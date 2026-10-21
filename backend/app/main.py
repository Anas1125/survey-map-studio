from __future__ import annotations

import io, json, os, re, uuid, time, resource, hashlib
import logging
import sys
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Literal

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from .manual_edits import apply_manual_operations, display_to_source_point
from .shape_analysis import detect_box_symbols, detect_dashed_guides, build_guide_removal_mask, build_blue_symbol_mask

APP_DIR=Path(__file__).resolve().parent
logger=logging.getLogger('survey_map_studio.cleaning')
DATA_DIR=Path(os.getenv('SURVEY_MAP_DATA_DIR', APP_DIR.parent/'data')).resolve()
ORIGINALS_DIR=DATA_DIR/'originals'; PROCESSED_DIR=DATA_DIR/'processed'; THUMB_DIR=DATA_DIR/'thumbnails'; PROJECTS_DIR=DATA_DIR/'projects'; OCR_CACHE_DIR=DATA_DIR/'ocr_cache'; MANUAL_EDITS_DIR=DATA_DIR/'manual_edits'; JOBS_DIR=DATA_DIR/'jobs'
for d in (ORIGINALS_DIR,PROCESSED_DIR,THUMB_DIR,PROJECTS_DIR,OCR_CACHE_DIR,MANUAL_EDITS_DIR,JOBS_DIR): d.mkdir(parents=True,exist_ok=True)
MAX_UPLOAD_BYTES=40*1024*1024
MAX_CONCURRENT_CLEANING=max(1,int(os.getenv('SURVEY_MAP_CLEANING_WORKERS','1')))
CLEANING_SLOTS=BoundedSemaphore(MAX_CONCURRENT_CLEANING)
ALLOWED_SUFFIXES={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}
app=FastAPI(title='Survey Map Studio API',version='0.2.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])
lock=Lock(); assets:dict[str,dict]={}; jobs:dict[str,dict]={}; projects:dict[str,dict]={}; manual_locks:dict[str,Lock]={}
OCR_EXECUTOR=ThreadPoolExecutor(max_workers=2,thread_name_prefix='survey-ocr')
def now_iso(): return datetime.now(timezone.utc).isoformat()
def _atomic_json(path:Path,data):
    temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(data)); temporary.replace(path)
def persist_job(job_id):
    with lock: data={k:v for k,v in jobs.get(job_id,{}).items() if not k.startswith('_')}
    if data: _atomic_json(JOBS_DIR/f'{job_id}.json',data)
def persist_assets():
    with lock: data=list(assets.values())
    _atomic_json(DATA_DIR/'asset_catalog.json',data)

def make_thumbnail(source:Path, asset_id:str):
    target=THUMB_DIR/f'{asset_id}.png'; temporary=target.with_suffix('.png.tmp')
    try:
        with Image.open(source) as image:
            image=ImageOps.exif_transpose(image).convert('RGB')
            image.thumbnail((480,320))
            image.save(temporary,format='PNG')
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

def normalize_asset_record(raw):
    """Repair recoverable catalog metadata without guessing user-facing metadata."""
    if not isinstance(raw,dict):
        return {'id':None,'filename':None,'map_type':None,'width':None,'height':None,'uploaded_at':None,'source_path':None,'_metadata_warnings':['Catalog entry is not an object.']}
    asset=dict(raw); warnings=[]
    asset_id=asset.get('id')
    if not isinstance(asset_id,str) or not asset_id.strip():
        warnings.append('Stable asset ID is missing; this catalog record cannot be linked to an image automatically.')
        asset_id=None
    else: asset_id=asset_id.strip()
    source=None
    given=asset.get('source_path')
    if isinstance(given,str) and given:
        candidate=Path(given)
        if candidate.is_file(): source=candidate.resolve()
    if source is None and asset_id:
        matches=[p for p in ORIGINALS_DIR.glob(f'{asset_id}.*') if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES]
        if len(matches)==1: source=matches[0].resolve()
        elif len(matches)>1: warnings.append('Multiple original files match this asset ID; source path was not guessed.')
    if source:
        asset['source_path']=str(source)
        try:
            with Image.open(source) as image: actual_width,actual_height=image.size
            if asset.get('width')!=actual_width or asset.get('height')!=actual_height:
                warnings.append('Image dimensions were restored from the original file.')
            asset['width']=actual_width; asset['height']=actual_height
        except (OSError,UnidentifiedImageError,ValueError):
            warnings.append('Original file exists but its image metadata could not be read.')
        if asset_id:
            thumb=THUMB_DIR/f'{asset_id}.png'
            valid_thumb=False
            if thumb.is_file():
                try:
                    with Image.open(thumb) as image: image.verify()
                    valid_thumb=True
                except (OSError,UnidentifiedImageError,ValueError): pass
            if not valid_thumb:
                try: make_thumbnail(source,asset_id)
                except (OSError,UnidentifiedImageError,ValueError) as exc:
                    warnings.append(f'Thumbnail could not be generated: {type(exc).__name__}.')
    else:
        asset['source_path']=None
        warnings.append('Original file could not be located; the catalog record was retained.')
    if not isinstance(asset.get('filename'),str) or not asset.get('filename','').strip():
        asset['filename']=None
        warnings.append('Original upload filename is unavailable; it was not inferred from the storage name.')
    if asset.get('map_type') not in {'primary','sheet'}:
        asset['map_type']=None
        warnings.append('Map type is unavailable; classify this map in the workspace when needed.')
    for dimension in ('width','height'):
        value=asset.get(dimension)
        if not isinstance(value,int) or isinstance(value,bool) or value<=0:
            asset[dimension]=None
            warnings.append(f'{dimension.capitalize()} is unavailable or invalid.')
    if not isinstance(asset.get('uploaded_at'),str) or not asset.get('uploaded_at','').strip():
        asset['uploaded_at']=None
    if asset_id is not None: asset['id']=asset_id
    asset['_metadata_warnings']=list(dict.fromkeys(warnings))
    return asset

@app.on_event('startup')
def restore_persisted_state():
    catalog=DATA_DIR/'asset_catalog.json'
    catalog_changed=False
    if catalog.exists():
        try:
            records=json.loads(catalog.read_text())
            if not isinstance(records,list): raise ValueError('Asset catalog must contain a list.')
            for index,raw in enumerate(records):
                asset=normalize_asset_record(raw); key=asset.get('id') or f'_unaddressable_{index}'
                assets[key]=asset
                catalog_changed |= asset!=raw
        except (OSError,ValueError,TypeError): logger.exception('Could not restore the local asset catalog')
    # Previous catalog writes could omit records while original TIFFs remained intact.
    # Recreate addressable records from storage IDs, leaving unknown names/types blank.
    for source in ORIGINALS_DIR.iterdir():
        if not source.is_file() or source.suffix.lower() not in ALLOWED_SUFFIXES: continue
        if source.stem in assets: continue
        recovered=normalize_asset_record({'id':source.stem,'source_path':str(source)})
        assets[source.stem]=recovered; catalog_changed=True
        logger.warning('Recovered asset %s from its original file; upload filename and map type are unavailable.',source.stem)
    if catalog_changed: persist_assets()
    for path in JOBS_DIR.glob('*.json'):
        try:
            job=json.loads(path.read_text())
            if job.get('status') in {'queued','processing'}:
                job.update(status='failed',stage='failed',message='Processing stopped when the API restarted. Retry this source.',error='Interrupted by API restart.')
            jobs[job['id']]=job
            persist_job(job['id'])
        except (OSError,ValueError,KeyError): logger.exception('Could not restore persisted job %s',path.stem)
def read_rgb(path):
    with Image.open(path) as im: return np.array(ImageOps.exif_transpose(im).convert('RGB'))
def encode_png(rgb):
    ok,b=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    if not ok: raise ValueError('PNG encoding failed')
    return b.tobytes()
def _cache_path(cache_key, detector):
    token=hashlib.sha256(f'{detector}:{cache_key}'.encode()).hexdigest()
    return OCR_CACHE_DIR/f'{token}.json'
def _read_detection_cache(cache_key, detector):
    if not cache_key: return None
    try: return json.loads(_cache_path(cache_key,detector).read_text())
    except (OSError,ValueError): return None
def _write_detection_cache(cache_key, detector, value):
    if not cache_key: return
    target=_cache_path(cache_key,detector); temporary=target.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(value)); temporary.replace(target)
    except OSError:
        temporary.unlink(missing_ok=True)
def apply_removal_mask(rgb:np.ndarray, mask:np.ndarray, *, dilation:int=2, protected:np.ndarray|None=None)->np.ndarray:
    """Inpaint exactly the supplied source-coordinate mask; never restore masked ink afterward."""
    if rgb.shape[:2] != mask.shape[:2]:
        raise ValueError(f'Mask/image dimensions differ: {mask.shape[:2]} vs {rgb.shape[:2]}')
    expanded=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,2*dilation+1)),iterations=1) if dilation else mask.astype(np.uint8)
    if protected is not None:
        if protected.shape != mask.shape: raise ValueError('Protected-label mask dimensions do not match image.')
        expanded[protected>0]=0
    result=cv2.inpaint(rgb,expanded,4,cv2.INPAINT_TELEA) if cv2.countNonZero(expanded) else rgb.copy()
    if protected is not None: result[protected>0]=rgb[protected>0]
    return result

def build_principal_label_mask(shape, detections:list[dict], merged_labels:list[dict], padding:int=2,
                               rgb:np.ndarray|None=None, competing_boxes:list[dict]|None=None)->np.ndarray:
    """Protect OCR-confirmed standalone integer labels and integers paired with decimals.

    Integer content is deliberately preserved regardless of its value. A measurement
    must contain a decimal separator before it can enter the automatic removal masks.
    """
    height,width=shape[:2]; protected=np.zeros((height,width),np.uint8); boxes=[]
    for item in detections:
        if re.fullmatch(r'\d{1,4}',str(item.get('text','')).strip()): boxes.append(item)
    boxes.extend(item['protected_box'] for item in merged_labels if isinstance(item.get('protected_box'),dict))
    for box in boxes:
        x,y,w,h=[int(box.get(k,0)) for k in ('x','y','width','height')]
        if w<=0 or h<=0: continue
        x0=max(0,x-padding); y0=max(0,y-padding); x1=min(width,x+w+padding); y1=min(height,y+h+padding)
        if x1<=x0 or y1<=y0: continue
        if rgb is None:
            # Keep the legacy behavior for callers without source pixels.
            cv2.rectangle(protected,(x0,y0),(x1-1,y1-1),255,-1)
            continue
        if rgb.shape[:2] != (height,width): raise ValueError('Label image and protection mask dimensions differ.')
        # Protect the confirmed label's ink, not the complete OCR rectangle. This
        # keeps a nearby decimal removable when OCR boxes touch, while preserving
        # the integer glyph and its anti-aliased edge through all removal masks.
        crop=rgb[y0:y1,x0:x1]
        gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        chroma=crop.max(axis=2).astype(np.int16)-crop.min(axis=2).astype(np.int16)
        ink=((gray<238)|(chroma>=30)).astype(np.uint8)
        # OCR boxes can overlap even when the glyphs are adjacent. In the
        # overlap, assign ink to the closer recognized text region by its center;
        # this keeps the integer's last digit while leaving neighboring decimal
        # glyphs eligible for removal.
        yy,xx=np.mgrid[y0:y1,x0:x1]
        label_cx=x+w/2; label_cy=y+h/2
        label_distance=(xx-label_cx)**2+(yy-label_cy)**2
        for other in competing_boxes or []:
            ox,oy,ow,oh=[int(other.get(k,0)) for k in ('x','y','width','height')]
            if ow<=0 or oh<=0: continue
            if min(x+w,ox+ow)<=max(x,ox) or min(y+h,oy+oh)<=max(y,oy): continue
            other_distance=(xx-(ox+ow/2))**2+(yy-(oy+oh/2))**2
            ink[other_distance<label_distance]=0
        if cv2.countNonZero(ink):
            # Include the annotation-mask padding and the later inpaint dilation,
            # so the edge of a protected integer cannot be clipped by a neighbor.
            ink=cv2.dilate(ink,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
            protected[y0:y1,x0:x1][ink>0]=255
    return protected

def exclude_protected_pixels(mask:np.ndarray, protected:np.ndarray)->np.ndarray:
    """Return a mask with all OCR-confirmed label and geometry pixels excluded."""
    if mask.shape!=protected.shape: raise ValueError('Removal and protected masks must share image dimensions.')
    result=mask.copy(); result[protected>0]=0
    return result

def _red_ink_fraction(rgb:np.ndarray, box:dict)->float:
    """Measure saturated red ink in an OCR box for color-aware label protection."""
    x,y,w,h=[int(box.get(k,0)) for k in ('x','y','width','height')]
    x0=max(0,x); y0=max(0,y); x1=min(rgb.shape[1],x+w); y1=min(rgb.shape[0],y+h)
    if x1<=x0 or y1<=y0: return 0.0
    hsv=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2HSV)
    red=cv2.bitwise_or(cv2.inRange(hsv,np.array([0,30,45],np.uint8),np.array([16,255,255],np.uint8)),cv2.inRange(hsv,np.array([164,30,45],np.uint8),np.array([179,255,255],np.uint8)))
    return float(cv2.countNonZero(red))/max(1,(x1-x0)*(y1-y0))

def is_principal_integer(rgb:np.ndarray, box:dict)->bool:
    return bool(re.fullmatch(r'\d{1,4}',str(box.get('text','')).strip()) and _red_ink_fraction(rgb,box)<.12)

def _is_decimal_ocr_alias(integer_box: dict, decimal_boxes: list[dict]) -> bool:
    """Ignore a weak integer OCR reading that duplicates a decimal's same glyph region."""
    if not re.fullmatch(r"\d{1,4}", str(integer_box.get("text", "")).strip()):
        return False

    raw_confidence = integer_box.get("confidence")
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else 100.0
    except (TypeError, ValueError):
        return False

    x, y, w, h = (
        int(integer_box.get(k, 0)) for k in ("x", "y", "width", "height")
    )
    if w <= 0 or h <= 0:
        return False

    for decimal in decimal_boxes:
        if not re.search(r"\d\s*[.,]\s*\d", str(decimal.get("text", ""))):
            continue

        dx, dy, dw, dh = (
            int(decimal.get(k, 0)) for k in ("x", "y", "width", "height")
        )
        if dw <= 0 or dh <= 0:
            continue

        ix = max(0, min(x + w, dx + dw) - max(x, dx))
        iy = max(0, min(y + h, dy + dh) - max(y, dy))
        overlap = (ix * iy) / max(1, min(w * h, dw * dh))

        distance = float(np.hypot(
            x + w / 2 - (dx + dw / 2),
            y + h / 2 - (dy + dh / 2),
        ))

        # Strongly contained OCR boxes can represent individual digits of
        # the same decimal, even when their centres are far apart.
        if overlap >= 0.80 or (
            overlap >= 0.75 and distance <= 0.25 * min(h, dh)
        ):
            return True

    return False

def build_red_ink_mask(rgb:np.ndarray)->np.ndarray:
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    return cv2.bitwise_or(cv2.inRange(hsv,np.array([0,30,45],np.uint8),np.array([16,255,255],np.uint8)),cv2.inRange(hsv,np.array([164,30,45],np.uint8),np.array([179,255,255],np.uint8)))
def map_ocr_box(x:int,y:int,w:int,h:int,rotation:int,width:int,height:int,scale:int=1):
    """Map a Tesseract rectangle from a rotated/scaled view into source coordinates."""
    x//=scale; y//=scale; w=max(1,w//scale); h=max(1,h//scale)
    if rotation==90: x,y,w,h=width-y-h,x,h,w
    elif rotation==270: x,y,w,h=y,height-x-w,h,w
    elif rotation==180: x,y=width-x-w,height-y-h
    x=max(0,min(width-1,x)); y=max(0,min(height-1,y))
    return x,y,max(0,min(width-x,w)),max(0,min(height-y,h))
def _ocr_small_region(rgb,x0,y0,x1,y1):
    import pytesseract
    crop=rgb[max(0,y0):min(rgb.shape[0],y1),max(0,x0):min(rgb.shape[1],x1)]
    if crop.size==0: return '',0
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY); _,bw=cv2.threshold(gray,168,255,cv2.THRESH_BINARY)
    bw=cv2.copyMakeBorder(bw,16,16,16,16,cv2.BORDER_CONSTANT,value=255)
    bw=cv2.resize(bw,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
    data=pytesseract.image_to_data(bw,output_type=pytesseract.Output.DICT,config='--psm 7 -c tessedit_char_whitelist=0123456789.,')
    words=[(t.strip(),float(data['conf'][i])) for i,t in enumerate(data['text']) if t.strip()]
    return ''.join(t for t,_ in words), max((c for _,c in words),default=-1)

def _ocr_measurement_crop(rgb, config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.,()'):
    """OCR a tightly proposed text line. Blue isolation keeps thick black parcel lines out."""
    import pytesseract
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    blue=cv2.inRange(hsv,np.array([90,45,35],np.uint8),np.array([145,255,255],np.uint8))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Preserve blue ink as black; this avoids grayscale loss of blue against white.
    # For black-only crops the isolation image is entirely white and an OCR call
    # against it cannot find the label, so skip that pass instead of paying for a
    # guaranteed miss before the unchanged grayscale/adaptive fallbacks.
    has_blue=cv2.countNonZero(blue)>=3
    variants=[]
    if has_blue: variants.append(np.where(blue>0,0,255).astype(np.uint8))
    otsu_index=len(variants)
    variants.append(cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
    adaptive_index=None
    if min(gray.shape)>=15:
        adaptive_index=len(variants)
        variants.append(cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,5))
    # Keep the same OCR scales and page-segmentation fallbacks, but only execute
    # the color-isolation pass when the crop actually contains blue annotation ink.
    attempts=[]
    if has_blue: attempts.append((0,3,7))
    attempts.append((otsu_index,3,7))
    if adaptive_index is not None: attempts.append((adaptive_index,4,7))
    attempts.append((otsu_index,4,8))
    if adaptive_index is not None: attempts.append((adaptive_index,4,13))
    for variant_index,scale,psm in attempts:
        if variant_index>=len(variants): continue
        variant=variants[variant_index]
        padded=cv2.copyMakeBorder(variant,16,16,16,16,cv2.BORDER_CONSTANT,value=255)
        enlarged=cv2.resize(padded,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC if scale==4 else cv2.INTER_NEAREST)
        active_config=config.replace('--psm 7',f'--psm {psm}')
        text=pytesseract.image_to_string(enlarged,config=active_config).strip().replace('\n','').replace(' ','')
        normalized=text.replace(',', '.')
        if re.fullmatch(r'[([]?\d{1,4}\.\d{1,3}[)\]]?',normalized): return normalized,100.0
    return '',-1.0

def _has_decimal_component_signature(components:list[tuple])->bool:
    """Cheap punctuation-shape prefilter before expensive OCR crop retries."""
    if len(components)<3: return False
    heights=np.asarray([c[3] for c in components],dtype=np.float32)
    areas=np.asarray([c[4] for c in components],dtype=np.float32)
    typical_height=float(np.median(heights)); typical_area=float(np.median(areas))
    if typical_height<3 or typical_area<1: return False
    larger=[c for c in components if c[4]>typical_area*.55 or max(c[2],c[3])>typical_height*.62]
    for small in components:
        if not (small[4]<=typical_area*.55 and small[2]<=max(2,typical_height*.62) and small[3]<=max(2,typical_height*.62)): continue
        center=small[0]+small[2]/2
        left=any(0<=center-(c[0]+c[2])<=typical_height*.8 for c in larger)
        right=any(0<=c[0]-center<=typical_height*.8 for c in larger)
        if left and right: return True
    return False

def _ocr_blue_integer_crop(rgb:np.ndarray)->tuple[str,float]:
    """Read blue parcel-number glyphs while suppressing black parcel lines below them."""
    import pytesseract
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    blue=cv2.inRange(hsv,np.array([82,38,32],np.uint8),np.array([153,255,255],np.uint8))
    if cv2.countNonZero(blue)<5: return '',-1.0
    isolated=np.where(blue>0,0,255).astype(np.uint8)
    padded=cv2.copyMakeBorder(isolated,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
    enlarged=cv2.resize(padded,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
    data=pytesseract.image_to_data(enlarged,output_type=pytesseract.Output.DICT,config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789')
    tokens=[(str(text).strip(),float(data['conf'][i])) for i,text in enumerate(data['text']) if str(text).strip()]
    return ''.join(text for text,_ in tokens),max((confidence for _,confidence in tokens),default=-1.0)

def detect_color_measurements(rgb:np.ndarray,tile_size:int=2048,*,cache_key:str|None=None,include_labels:bool=False):
    """Propose decimal text lines from blue or black ink components, then validate with Tesseract.

    This targets sparse colored measurements missed by full-page grayscale OCR. Only OCR
    validated decimals are returned; integer plot IDs are never candidates. Rotated views
    are handled by rotating each tile before segmentation and mapping boxes back exactly.
    """
    cached=_read_detection_cache(cache_key,'color-measurements')
    cached_labels=_read_detection_cache(cache_key,'color-integer-labels') if include_labels else None
    if cached is not None and (not include_labels or cached_labels is not None):
        return (cached,cached_labels) if include_labels else cached
    H,W=rgb.shape[:2]; found=[]; labels_found=[]; pending_ocr=[]; pending_by_key={}; seen=set(); seen_regions=set(); seen_label_regions=set(); overlap=80
    def enqueue_ocr(kind,region_key,runner,crop,box,ink):
        key=(kind,region_key); payload=(crop.copy(),box,ink)
        existing=pending_by_key.get(key)
        if existing is not None:
            # Preserve the original orientation retry order, but defer alternates
            # until the first crop fails. This avoids duplicate OCR work while
            # the first attempt runs concurrently with later tile analysis.
            existing['alternatives'].append(payload)
            return
        entry={'kind':kind,'region_key':region_key,'runner':runner,'box':box,'ink':ink,
               'future':OCR_EXECUTOR.submit(runner,payload[0]),'alternatives':[]}
        pending_by_key[key]=entry; pending_ocr.append(entry)
    # Connected components are unchanged by a 180° rotation. Reuse the 0°/90°
    # component boxes in the opposite directions and rotate only candidate crops
    # for OCR, avoiding two redundant full-tile color/threshold/label passes.
    components_by_tile_rotation={}
    for rotation in (0,90,270,180):
        for y0 in range(0,H,tile_size-overlap):
            y1=min(H,y0+tile_size)
            for x0 in range(0,W,tile_size-overlap):
                x1=min(W,x0+tile_size)
                tile=rgb[y0:y1,x0:x1]
                view=np.rot90(tile,rotation//90).copy() if rotation else tile
                if rotation in (0,90):
                    hsv=cv2.cvtColor(view,cv2.COLOR_RGB2HSV)
                    gray=cv2.cvtColor(view,cv2.COLOR_RGB2GRAY)
                    blue_mask=cv2.inRange(hsv,np.array([90,45,35],np.uint8),np.array([145,255,255],np.uint8))
                    black_mask=cv2.threshold(gray,115,255,cv2.THRESH_BINARY_INV)[1]
                    otsu_black=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
                    black_mask=cv2.bitwise_or(black_mask,otsu_black)
                    black_mask[blue_mask>0]=0
                    components_for_ink={}
                    for ink,component_mask in (('blue',blue_mask),('black',black_mask)):
                        n,labels,stats,_=cv2.connectedComponentsWithStats(component_mask,8)
                        comps=[]
                        for i in range(1,n):
                            cx,cy,cw,ch,area=map(int,stats[i])
                            if area<1 or area>5000 or ch<1 or ch>140 or cw>120: continue
                            comps.append((cx,cy,cw,ch,area))
                        components_for_ink[ink]=comps
                else:
                    base_rotation=rotation-180
                    base_width=tile.shape[1] if base_rotation==0 else tile.shape[0]
                    base_height=tile.shape[0] if base_rotation==0 else tile.shape[1]
                    base_components=components_by_tile_rotation[(y0,x0,base_rotation)]
                    components_for_ink={ink:[(base_width-cx-cw,base_height-cy-ch,cw,ch,area)
                                             for cx,cy,cw,ch,area in base]
                                        for ink,base in base_components.items()}
                if rotation in (0,90): components_by_tile_rotation[(y0,x0,rotation)]=components_for_ink
                for ink,comps in components_for_ink.items():
                    # Compose glyph components into baseline-aligned line proposals.
                    unused=set(range(len(comps)))
                    while unused:
                        # Seed from a full-height glyph so the smaller decimal point
                        # is attached to its digit train instead of being isolated first.
                        seed=min(unused,key=lambda i:(-comps[i][3],comps[i][1]+comps[i][3],comps[i][0]))
                        group=[seed]; unused.remove(seed); changed=True
                        while changed:
                            changed=False; g=[comps[i] for i in group]
                            gx0=min(c[0] for c in g); gx1=max(c[0]+c[2] for c in g)
                            gy0=min(c[1] for c in g); gy1=max(c[1]+c[3] for c in g)
                            medh=float(np.median([c[3] for c in g]))
                            for j in list(unused):
                                x,y,w,h,a=comps[j]; gap=max(0,x-gx1,gx0-(x+w))
                                baseline=abs((y+h)-gy1)
                                height_ratio=h/max(1,medh)
                                punctuation=height_ratio<.45 and baseline<=max(10,medh*.55)
                                if gap<=max(18,medh*.7) and (baseline<=max(10,medh*.28) or punctuation) and (.12<=height_ratio<=1.8):
                                    group.append(j); unused.remove(j); changed=True
                        g=[comps[i] for i in group]
                        if not g: continue
                        bx0=min(c[0] for c in g); by0=min(c[1] for c in g); bx1=max(c[0]+c[2] for c in g); by1=max(c[1]+c[3] for c in g)
                        if bx1-bx0>500 or by1-by0>150 or bx1-bx0<3: continue
                        decimal_signature=_has_decimal_component_signature(g)
                        if not decimal_signature:
                            # A crop-level pass recovers small blue parcel IDs omitted
                            # by sparse page OCR. It only protects OCR-confirmed integers.
                            if not include_labels or ink!='blue' or not (8<=by1-by0<=120 and bx1-bx0<=max(30,(by1-by0)*2.5) and len(g)<=4): continue
                            rx,ry,rw,rh=map_ocr_box(bx0,by0,bx1-bx0,by1-by0,rotation,tile.shape[1],tile.shape[0]); rx+=x0; ry+=y0
                            label_key=(rx//4,ry//4,rw//4,rh//4)
                            crop=view[
                                max(0,by0-8):min(view.shape[0],by1+8),
                                max(0,bx0-8):min(view.shape[1],bx1+8)
                            ]

                            enqueue_ocr(
                                'label',
                                label_key,
                                _ocr_blue_integer_crop,
                                crop,
                                (rx,ry,rw,rh),
                                ink
                            )
                            continue
                        rx,ry,rw,rh=map_ocr_box(bx0,by0,bx1-bx0,by1-by0,rotation,tile.shape[1],tile.shape[0])
                        rx+=x0; ry+=y0
                        region_key=(rx//4,ry//4,rw//4,rh//4)
                        crop=view[max(0,by0-8):min(view.shape[0],by1+8),max(0,bx0-8):min(view.shape[1],bx1+8)]
                        enqueue_ocr('measurement',region_key,_ocr_measurement_crop,crop,(rx,ry,rw,rh),ink)
    # Resolve in original candidate order, preserving the prior rule: a second
    # orientation may retry a region only if earlier OCR failed to validate it.
    for entry in pending_ocr:
        kind=entry['kind']; region_key=entry['region_key']; ink=entry['ink']
        result=entry['future'].result(); box=entry['box']
        valid=(re.fullmatch(r'\d{1,4}',result[0]) is not None) if kind=='label' else bool(result[0])
        for alternate_crop,alternate_box,alternate_ink in entry['alternatives']:
            if valid: break
            result=OCR_EXECUTOR.submit(entry['runner'],alternate_crop).result(); box=alternate_box; ink=alternate_ink
            valid=(re.fullmatch(r'\d{1,4}',result[0]) is not None) if kind=='label' else bool(result[0])
        if not valid: continue
        rx,ry,rw,rh=box
        if kind=='label':
            if region_key in seen_label_regions: continue
            label_text,label_conf=result
            seen_label_regions.add(region_key)
            labels_found.append({'text':label_text,'x':rx,'y':ry,'width':rw,'height':rh,'confidence':label_conf,'kind':'principal-label','color':ink,'detection_method':'blue-component-integer-ocr'})
            continue
        if region_key in seen_regions: continue
        text,conf=result
        seen_regions.add(region_key)
        # OCR crop includes an 8px margin; return the component-derived tight box.
        if rw<1 or rh<1 or rx+rw>W or ry+rh>H: continue
        key=(text,rx//5,ry//5,rw//5,rh//5)
        if key in seen: continue
        seen.add(key)
        found.append({'text':text,'x':rx,'y':ry,'width':rw,'height':rh,'confidence':conf,'kind':'measurement','detection_method':f'{ink}-component-line-ocr'})
    _write_detection_cache(cache_key,'color-measurements',found)
    if include_labels:
        _write_detection_cache(cache_key,'color-integer-labels',labels_found)
        return found,labels_found
    return found

def _rotate_expand(gray:np.ndarray,angle:float,border:int=255)->np.ndarray:
    h,w=gray.shape[:2]; rad=np.deg2rad(angle)
    nw=int(abs(h*np.sin(rad))+abs(w*np.cos(rad)))+2; nh=int(abs(h*np.cos(rad))+abs(w*np.sin(rad)))+2
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0); matrix[0,2]+=nw/2-w/2; matrix[1,2]+=nh/2-h/2
    return cv2.warpAffine(gray,matrix,(nw,nh),flags=cv2.INTER_NEAREST,borderValue=border)

def _ocr_oriented_measurement(rgb:np.ndarray,angle:float,ink:str):
    """Deskew and OCR a color-isolated measurement crop at its native source resolution."""
    import pytesseract
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if ink=='blue': mask=cv2.inRange(hsv,np.array([90,45,35],np.uint8),np.array([145,255,255],np.uint8))
    elif ink=='red': mask=cv2.bitwise_or(cv2.inRange(hsv,np.array([0,55,55],np.uint8),np.array([12,255,255],np.uint8)),cv2.inRange(hsv,np.array([168,55,55],np.uint8),np.array([179,255,255],np.uint8)))
    else: mask=cv2.threshold(gray,115,255,cv2.THRESH_BINARY_INV)[1]
    isolated=np.where(mask>0,0,255).astype(np.uint8)
    fallback=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    configs = [
        (angle, isolated, 7, 2),
        (angle, isolated, 7, 3),
        (angle, isolated, 10, 2),
        (angle, isolated, 11, 2),
        (angle, fallback, 7, 2),
        (angle - 10, isolated, 7, 2),
        (angle + 10, isolated, 7, 2),
    ]

    # PCA identifies a line's axis, but either direction along that axis is valid.
    # If normal deskew attempts fail, try the opposite text orientation.
    opposite_angle = angle - 180 if angle >= 0 else angle + 180
    configs.extend([
        (opposite_angle, isolated, 7, 2),
        (opposite_angle - 10, isolated, 7, 2),
        (opposite_angle + 10, isolated, 7, 2),
    ])
    for theta,variant,psm,scale in configs:
        rotated=_rotate_expand(variant,theta)
        canvas=cv2.copyMakeBorder(rotated,18,18,18,18,cv2.BORDER_CONSTANT,value=255)
        enlarged=cv2.resize(canvas,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
        data=pytesseract.image_to_data(enlarged,output_type=pytesseract.Output.DICT,config=f'--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789.,()[]')
        tokens=[(t.strip(),float(data['conf'][i])) for i,t in enumerate(data['text']) if t.strip()]
        text=''.join(t for t,_ in tokens).replace(',','.').strip()
        text=text.strip('.,') if text not in ('',) else text
        if re.fullmatch(r'[([]?\d{1,4}\.\d{1,3}[)\]]?',text):
            return text,max((c for _,c in tokens),default=70.0)
    return '',-1.0

def detect_oriented_measurements(rgb:np.ndarray,tile_size:int=3072,*,cache_key:str|None=None)->list[dict]:
    """Find diagonal decimal labels by clustering color glyph components and deskewing crops.

    Candidate text is OCR-validated before it is returned. Component-level boxes are retained
    for tight masking, while the axis-aligned source bbox provides review coordinates.
    """
    cached=_read_detection_cache(cache_key,'oriented-measurements')
    if cached is not None: return cached
    H,W=rgb.shape[:2]; found=[]; seen=set(); overlap=min(700,max(160,tile_size//4)); step=tile_size-overlap
    for y0 in range(0,H,step):
        y1=min(H,y0+tile_size)
        for x0 in range(0,W,step):
            x1=min(W,x0+tile_size); tile=rgb[y0:y1,x0:x1]
            hsv=cv2.cvtColor(tile,cv2.COLOR_RGB2HSV); gray=cv2.cvtColor(tile,cv2.COLOR_RGB2GRAY)
            blue=cv2.inRange(hsv,np.array([90,45,35],np.uint8),np.array([145,255,255],np.uint8))
            red=cv2.bitwise_or(cv2.inRange(hsv,np.array([0,55,55],np.uint8),np.array([12,255,255],np.uint8)),cv2.inRange(hsv,np.array([168,55,55],np.uint8),np.array([179,255,255],np.uint8)))
            black=cv2.threshold(gray,115,255,cv2.THRESH_BINARY_INV)[1]; black[(blue>0)|(red>0)]=0
            for ink,ink_mask in (('blue',blue),('red',red),('black',black)):
                n,_,stats,_=cv2.connectedComponentsWithStats(ink_mask,8); comps=[]
                for i in range(1,n):
                    x,y,w,h,a=map(int,stats[i])
                    if a<1 or a>6000 or w<1 or h<2 or w>130 or h>130: continue
                    comps.append((x,y,w,h,a,x+w/2,y+h/2))
                if len(comps)<3: continue
                # Spatial buckets avoid a quadratic all-pairs scan on dense survey sheets.
                buckets={}
                for i,c in enumerate(comps): buckets.setdefault((int(c[5]//80),int(c[6]//80)),[]).append(i)
                parent=list(range(len(comps)))
                def root(i):
                    while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
                    return i
                def join(a,b):
                    ra,rb=root(a),root(b)
                    if ra!=rb: parent[rb]=ra
                for i,c in enumerate(comps):
                    gx,gy=int(c[5]//80),int(c[6]//80)
                    for xx in range(gx-1,gx+2):
                        for yy in range(gy-1,gy+2):
                            for j in buckets.get((xx,yy),()):
                                if j<=i: continue
                                d=np.hypot(c[5]-comps[j][5],c[6]-comps[j][6])
                                # Decimal dots and parentheses are much smaller than digits;
                                # keep them in the same glyph chain so the removal mask includes them.
                                ratio=max(c[2],c[3])/max(1,max(comps[j][2],comps[j][3]))
                                if d<=78 and .12<=ratio<=6.5: join(i,j)
                groups={}
                for i in range(len(comps)): groups.setdefault(root(i),[]).append(i)
                for indices in groups.values():
                    if len(indices)<2 or len(indices)>12: continue
                    areas=[comps[i][4] for i in indices]
                    typical_area=float(np.median(areas))
                    # Decimal measurement groups need a punctuation-sized component.
                    # Reject long parcel boundaries and ordinary integer labels before OCR.
                    if not any(comps[i][4]<typical_area*.42 and comps[i][2]<=18 and comps[i][3]<=18 for i in indices): continue
                    pts=np.array([[comps[i][5],comps[i][6]] for i in indices],np.float32)
                    mean,eigenvectors,_=cv2.PCACompute2(pts,mean=None)
                    angle=float(np.degrees(np.arctan2(eigenvectors[0,1],eigenvectors[0,0])))
                    angle=(angle+90)%180-90
                    median_h=float(np.median([max(comps[i][2],comps[i][3]) for i in indices]))
                    residual=np.abs((pts-mean)@eigenvectors[1])
                    if float(residual.max())>median_h*.48: continue
                    bx0=min(comps[i][0] for i in indices); by0=min(comps[i][1] for i in indices)
                    bx1=max(comps[i][0]+comps[i][2] for i in indices); by1=max(comps[i][1]+comps[i][3] for i in indices)
                    if bx1-bx0>650 or by1-by0>650: continue
                    edge=max(120,int(median_h*1.5))
                    if (x0>0 and bx0<edge) or (y0>0 and by0<edge) or (x1<W and tile.shape[1]-bx1<edge) or (y1<H and tile.shape[0]-by1<edge): continue
                    pad=max(18,int(median_h*.38)); cx0=max(0,bx0-pad); cy0=max(0,by0-pad); cx1=min(tile.shape[1],bx1+pad); cy1=min(tile.shape[0],by1+pad)
                    text,confidence=_ocr_oriented_measurement(tile[cy0:cy1,cx0:cx1],round(angle/5)*5,ink)
                    if not text: continue
                    boxes = [
                        {
                            'x': x0 + comps[i][0],
                            'y': y0 + comps[i][1],
                            'width': comps[i][2],
                            'height': comps[i][3],
                        }
                        for i in indices
                    ]

                    rx = x0 + bx0
                    ry = y0 + by0
                    rw = bx1 - bx0
                    rh = by1 - by0
                    key=(text,rx//5,ry//5,rw//5,rh//5)
                    if key in seen: continue
                    seen.add(key)
                    found.append({'text':text,'x':rx,'y':ry,'width':rw,'height':rh,'confidence':confidence,'kind':'measurement','detection_method':f'{ink}-oriented-component-ocr','component_boxes':boxes})
    _write_detection_cache(cache_key,'oriented-measurements',found)
    return found

def detect_merged_measurement_labels(rgb:np.ndarray,tile_size:int=2048,*,cache_key:str|None=None)->list[dict]:
    """Find a decimal measurement touching a larger, lower-baseline parcel identifier.

    Shape/baseline analysis identifies separate glyph groups; OCR validates both groups separately.
    Returns only the measurement box and separately records the protected integer label box.
    """
    cached=_read_detection_cache(cache_key,'merged-measurement-labels')
    if cached is not None: return cached
    import pytesseract  # noqa: F401 - fail visibly to the caller if OCR was configured incorrectly
    H,W=rgb.shape[:2]; found=[]; seen=set(); overlap=96
    for y0 in range(0,H,tile_size-overlap):
        y1=min(H,y0+tile_size)
        for x0 in range(0,W,tile_size-overlap):
            x1=min(W,x0+tile_size); tile=rgb[y0:y1,x0:x1]
            hsv=cv2.cvtColor(tile,cv2.COLOR_RGB2HSV)
            blue=cv2.inRange(hsv,np.array([90,45,35],np.uint8),np.array([145,255,255],np.uint8))
            n,labels,stats,_=cv2.connectedComponentsWithStats(blue,8)
            comps=[]
            for cid in range(1,n):
                x,y,w,h,a=map(int,stats[cid])
                if a<5 or a>5000 or w<2 or h<2 or w>150 or h>150: continue
                comps.append((x+x0,y+y0,w,h,a,cid))
            for dot in comps:
                dx,dy,dw,dh,da,_=dot
                if not (5<=da<=160 and dw<=18 and dh<=18 and .35<=dw/max(1,dh)<=1.8): continue
                baseline=dy+dh
                aligned=[c for c in comps if c[4]>70 and 16<=c[3]<=130 and c[2]<=100 and abs((c[1]+c[3])-baseline)<=8]
                left=[]; cursor=dx
                while len(left)<3:
                    prev=[c for c in aligned if 0<=cursor-(c[0]+c[2])<=24]
                    if not prev: break
                    c=max(prev,key=lambda q:q[0]); left.append(c); cursor=c[0]
                left=list(reversed(left))
                nxt=[c for c in aligned if 0<=c[0]-(dx+dw)<=24]
                if len(left)<2 or not nxt: continue
                digit=nxt[0]; dec_group=left+[dot,digit]
                end=max(c[0]+c[2] for c in dec_group)
                tails=[]; cursor=end; median_h=float(np.median([c[3] for c in dec_group if c is not dot]))
                while len(tails)<2:
                    possible=[c for c in comps if c not in dec_group and 0<=c[0]-cursor<=24 and c[3]>=median_h*1.15 and 8<=(c[1]+c[3])-baseline<=48]
                    if not possible: break
                    c=min(possible,key=lambda q:q[0]); tails.append(c); cursor=c[0]+c[2]
                if not tails: continue
                tail_h=float(np.median([c[3] for c in tails]))
                if tail_h<median_h*1.15: continue
                mx0=min(c[0] for c in dec_group); my0=min(c[1] for c in dec_group); mx1=max(c[0]+c[2] for c in dec_group); my1=max(c[1]+c[3] for c in dec_group)
                px0=min(c[0] for c in tails); py0=min(c[1] for c in tails); px1=max(c[0]+c[2] for c in tails); py1=max(c[1]+c[3] for c in tails)
                measurement,mc=_ocr_small_region(rgb,mx0-5,my0-8,mx1+5,my1+8)
                principal,pc=_ocr_small_region(rgb,px0-5,py0-8,px1+5,py1+8)
                if not re.fullmatch(r'\d{1,4}[.,]\d{1,2}',measurement) or not re.fullmatch(r'\d{1,2}',principal): continue
                key=(mx0//4,my0//4,mx1//4,my1//4,px0//4,py0//4)
                if key in seen: continue
                seen.add(key)
                found.append({'text':measurement,'x':mx0,'y':my0,'width':mx1-mx0,'height':my1-my0,'confidence':min(mc,pc),'kind':'measurement','protected_text':principal,'protected_box':{'x':px0,'y':py0,'width':px1-px0,'height':py1-py0},'detection_method':'separate-glyph-and-baseline-ocr'})
    _write_detection_cache(cache_key,'merged-measurement-labels',found)
    return found
def public_asset(a):
    normalized=normalize_asset_record(a); asset_id=normalized.get('id')
    urls={'image_url':None,'thumbnail_url':None,'preview_url':None}
    if asset_id and normalized.get('source_path'):
        urls={'image_url':f'/api/maps/{asset_id}/image','thumbnail_url':f'/api/maps/{asset_id}/thumbnail','preview_url':f'/api/maps/{asset_id}/preview'}
    return {k:normalized.get(k) for k in ('id','filename','map_type','width','height','uploaded_at')} | urls | {'original_available':bool(normalized.get('source_path')),'latest_job_id':normalized.get('latest_job_id'),'approved':normalized.get('approved',False),'metadata_warnings':normalized.get('_metadata_warnings',[])}
def update_job(jid,**kw):
    with lock:
        if jid in jobs: jobs[jid].update(kw,updated_at=now_iso())
    persist_job(jid)

def current_candidate_path(job_id, job=None):
    if job is None:
        with lock: job=jobs.get(job_id)
    return Path(job.get('candidate_path') or (PROCESSED_DIR/f'{job_id}.png')) if job else PROCESSED_DIR/f'{job_id}.png'

def set_stage(jid, name, progress, message):
    """Close the previous stage timer and publish real elapsed/runtime data."""
    now=time.perf_counter()
    with lock:
        job=jobs.get(jid)
        if not job: return
        timings=job.setdefault('stage_timings',{})
        previous=job.get('_stage_started')
        if previous:
            duration=round(now-previous['at'],3)
            timings[previous['name']]=round(timings.get(previous['name'],0.0)+duration,3)
            logger.info('Cleaning job %s stage %s completed in %.3fs',jid,previous['name'],duration)
        job['_stage_started']={'name':name,'at':now}
        created=job.get('_started_perf',now)
        job.update(stage=name,progress=progress,message=message,elapsed_seconds=round(now-created,1),updated_at=now_iso())
    persist_job(jid)

def finish_job_timing(jid):
    with lock:
        job=jobs.get(jid)
        if not job: return
        previous=job.get('_stage_started'); now=time.perf_counter()
        if previous:
            timings=job.setdefault('stage_timings',{})
            timings[previous['name']]=round(timings.get(previous['name'],0.0)+(now-previous['at']),3)
        job.pop('_stage_started',None)
        job['elapsed_seconds']=round(now-job.get('_started_perf',now),1)
        try: job['peak_memory_mb']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024),1)
        except Exception: job['peak_memory_mb']=None
    persist_job(jid)

def _clean_candidate(jid, aid, path):
    try:
        source_stat=Path(path).stat()
        detection_cache_key=f'{aid}:{source_stat.st_size}:{source_stat.st_mtime_ns}:detectors-v12'
        update_job(jid,status='processing')
        set_stage(jid,'loading',3,'Loading source image')
        rgb=read_rgb(path); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        set_stage(jid,'full_page_ocr',8,'Analyzing annotations with full-page OCR')
        # OCR is optional. Only confident decimal/small red text is removable; uncertainty remains visible.
        removable=[]; uncertain=[]; detected=[]; red_boxes=[]; ocr_warnings=[]
        try:
            import pytesseract
            # Use native-size whole-page passes. The prior implementation created a 2x
            # 19,834 x 28,050 image and OCRed each full page in four orientations; this duplicated
            # work and made Tesseract dominate the job. Rotated measurements are handled by
            # the overlapping, color-aware crop detector below, at their original resolution.
            variants=[gray,cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]]
            seen=set()
            for variant in variants:
                rotation=0; scale=1
                data=pytesseract.image_to_data(variant,output_type=pytesseract.Output.DICT,config='--psm 11')
                for i,raw in enumerate(data['text']):
                        txt=raw.strip(); conf=float(data['conf'][i]) if str(data['conf'][i]).replace('.','',1).isdigit() else -1
                        if not txt: continue
                        x,y,w,h=[int(data[k][i]) for k in ('left','top','width','height')]
                        # Map rotated/upscaled OCR rectangles back into source coordinates.
                        x,y,w,h=map_ocr_box(x,y,w,h,rotation,gray.shape[1],gray.shape[0],scale)
                        if w<2 or h<2: continue
                        normalized=re.sub(r'^[([{]|[)\]}]$','',txt).replace(',', '.')
                        decimal=bool(re.search(r'\d{1,4}[.,]\d{1,3}',txt))
                        decimal=decimal or bool(re.fullmatch(r'[([]?\d{1,4}[.,]\d{1,3}[)\]]?',txt))
                        key=(normalized,x//4,y//4,w//4,h//4)
                        if key in seen: continue
                        seen.add(key)
                        box={'text':txt,'x':x,'y':y,'width':w,'height':h,'confidence':conf,'kind':'measurement' if decimal else 'text'}
                        if decimal and conf>=40: removable.append(box)
                        elif decimal: uncertain.append(box)
                        elif re.fullmatch(r'\d{1,3}',txt): uncertain.append(dict(box,kind='possible-principal-label'))
                        detected.append(box)
        except Exception as exc:
            ocr_warnings.append(f'OCR pass failed: {type(exc).__name__}: {exc}')
        set_stage(jid,'rotated_cropped_ocr',30,'Checking rotated and cropped measurement labels')
        merged_labels=[]
        try:
            merged_labels=detect_merged_measurement_labels(rgb,cache_key=detection_cache_key)
            detected.extend(merged_labels)
        except Exception as exc:
            ocr_warnings.append(f'Adjacent measurement/label analysis failed: {type(exc).__name__}: {exc}')
        set_stage(jid,'color_segmentation',42,'Segmenting colored and dark text regions')
        color_measurements=[]
        try:
            color_measurements,blue_integer_labels=detect_color_measurements(rgb,cache_key=detection_cache_key,include_labels=True)
            detected.extend(blue_integer_labels)
            set_stage(jid,'rotated_cropped_ocr',47,'Validating rotated OCR measurement crops')
            oriented_measurements=detect_oriented_measurements(rgb,cache_key=detection_cache_key)
            color_measurements.extend(oriented_measurements)
            # The sparse full-page OCR remains authoritative where it already found a box.
            # Add only non-overlapping color detections, so independently detected labels remain intact.
            known=removable+merged_labels
            for candidate in color_measurements:
                overlaps=any(max(0,min(candidate['x']+candidate['width'],b['x']+b['width'])-max(candidate['x'],b['x'])) * max(0,min(candidate['y']+candidate['height'],b['y']+b['height'])-max(candidate['y'],b['y'])) > .35*candidate['width']*candidate['height'] for b in known)
                if not overlaps:
                    uncertain=[b for b in uncertain if not (b.get('kind') in {'measurement','measurement-overlaps-principal-label'} and max(0,min(candidate['x']+candidate['width'],b['x']+b['width'])-max(candidate['x'],b['x'])) * max(0,min(candidate['y']+candidate['height'],b['y']+b['height'])-max(candidate['y'],b['y'])) > .35*candidate['width']*candidate['height'])]
                    removable.append(candidate); detected.append(candidate); known.append(candidate)
        except Exception as exc:
            ocr_warnings.append(f'Color-aware measurement analysis failed: {type(exc).__name__}: {exc}')
        # Color segmentation finds red annotations even when OCR misses tiny glyphs.
        # Keep connected-component labels tile-local; a full 13,908 x 9,917 int32
        # label image alone consumes roughly 556 MB on this source.
        H,W=rgb.shape[:2]; tile=1024; overlap=48; step=tile-overlap; seen_red=set()
        for y0 in range(0,H,step):
            y1=min(H,y0+tile)
            for x0 in range(0,W,step):
                x1=min(W,x0+tile); hsv_tile=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2HSV)
                red=cv2.bitwise_or(cv2.inRange(hsv_tile,np.array([0,56,56],np.uint8),np.array([12,255,255],np.uint8)),cv2.inRange(hsv_tile,np.array([168,56,56],np.uint8),np.array([179,255,255],np.uint8)))
                count,labels,stats,_=cv2.connectedComponentsWithStats(red,8)
                for i in range(1,count):
                    x,y,w,h,area=map(int,stats[i])
                    if area<2: continue
                    # Assign each component to exactly one tile by its center; the overlap
                    # guarantees its full glyph is visible in that owning tile.
                    cx=x+w/2; cy=y+h/2; tw=x1-x0; th=y1-y0
                    if (x0>0 and cx<overlap/2) or (y0>0 and cy<overlap/2) or (x1<W and cx>=tw-overlap/2) or (y1<H and cy>=th-overlap/2): continue
                    key=(x0+x,y0+y,w,h)
                    if key in seen_red: continue
                    seen_red.add(key)
                    b={'x':x0+x,'y':y0+y,'width':w,'height':h,'kind':'red-annotation','confidence':1.0}
                    red_boxes.append(b); detected.append(b)
        set_stage(jid,'line_analysis',52,'Detecting measurement and line candidates')
        box_symbols=detect_box_symbols(rgb)
        periodic_guides=detect_dashed_guides(gray, rgb=rgb)
        # Conservative line candidate detector: only short, isolated Hough segments are flagged;
        # no lines are automatically erased, because geometry cannot be safely inferred generically.
        edges=cv2.Canny(gray,60,160); lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=35,minLineLength=8,maxLineGap=3)
        dashed=[]
        if lines is not None:
            for line in lines[:,0,:]:
                x1,y1,x2,y2=map(int,line); length=float(np.hypot(x2-x1,y2-y1))
                if 8<=length<=45: dashed.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'confidence':'review'})
        set_stage(jid,'mask_creation',62,'Creating full-resolution removal masks')
        protected_boxes=[b['protected_box'] for b in merged_labels if isinstance(b.get('protected_box'),dict)]
        decimal_boxes = [
            b for b in removable
            if re.search(r"\d\s*[.,]\s*\d", str(b.get("text", "")))
            and float(b.get("confidence", 0) or 0) >= 85
        ]

        integer_detections = [
            b for b in detected
            if is_principal_integer(rgb, b)
            and not _is_decimal_ocr_alias(b, decimal_boxes)
        ]
        principal_boxes=[]
        principal_boxes.extend(integer_detections)
        principal_boxes.extend(dict(b,text=b.get('protected_text','')) for b in protected_boxes)
        unique_principal_boxes=[]; seen_principal=set()
        for b in principal_boxes:
            key=tuple(int(b.get(k,0)) for k in ('x','y','width','height'))
            if key not in seen_principal: seen_principal.add(key); unique_principal_boxes.append(b)
        accepted=[]
        for b in removable:
            ambiguous=False
            for pbox in unique_principal_boxes:
                ix=max(0,min(b['x']+b['width'],pbox['x']+pbox['width'])-max(b['x'],pbox['x']))
                iy=max(0,min(b['y']+b['height'],pbox['y']+pbox['height'])-max(b['y'],pbox['y']))
                if ix<=0 or iy<=0: continue
                # Rectangular overlap alone is not ambiguity. Only near-coincident
                # text centers are withheld; separated glyph trains are resolved
                # by the pixel-level nearest-label mask above.
                distance=float(np.hypot((b['x']+b['width']/2)-(pbox['x']+pbox['width']/2),
                                        (b['y']+b['height']/2)-(pbox['y']+pbox['height']/2)))
                scale=max(1.0,min(b['height'],pbox['height']))
                if distance < .35*scale: ambiguous=True; break
            if ambiguous: uncertain.append(dict(b,kind='measurement-overlaps-principal-label'))
            else: accepted.append(b)
        measurement_mask=np.zeros(gray.shape,np.uint8); red_removal_mask=np.zeros(gray.shape,np.uint8)
        safe_principal_detections=principal_boxes[:]
        protected_mask=build_principal_label_mask(gray.shape,safe_principal_detections,merged_labels,padding=2,
                                                  rgb=rgb,competing_boxes=accepted+merged_labels)
        for b in accepted+merged_labels:
            pad=1 if b in merged_labels else 2
            components=b.get('component_boxes')
            if components:
                for cbox in components:
                    cv2.rectangle(measurement_mask,(max(0,cbox['x']-pad),max(0,cbox['y']-pad)),(min(gray.shape[1]-1,cbox['x']+cbox['width']+pad),min(gray.shape[0]-1,cbox['y']+cbox['height']+pad)),255,-1)
            else:
                cv2.rectangle(measurement_mask,(max(0,b['x']-pad),max(0,b['y']-pad)),(min(gray.shape[1]-1,b['x']+b['width']+pad),min(gray.shape[0]-1,b['y']+b['height']+pad)),255,-1)
        # Red components are masked by their actual chromatic pixels, not filled
        # bounding boxes. This prevents red labels from pulling nearby black parcel
        # edges into an inpainting region.
        red_removal_mask=build_red_ink_mask(rgb)
        guide_removal_mask,boundary_protected_mask,periodic_guides=build_guide_removal_mask(gray,periodic_guides,rgb,protected_mask)
        all_protected_mask=cv2.bitwise_or(protected_mask,boundary_protected_mask)
        measurement_mask=exclude_protected_pixels(measurement_mask,all_protected_mask)
        red_removal_mask=exclude_protected_pixels(red_removal_mask,all_protected_mask)
        annotation_mask=cv2.bitwise_or(measurement_mask,red_removal_mask)
        guide_removal_mask=exclude_protected_pixels(guide_removal_mask,all_protected_mask)
        symbol_removal_mask,box_symbols=build_blue_symbol_mask(rgb,box_symbols,all_protected_mask)
        structure_mask=cv2.bitwise_or(guide_removal_mask,symbol_removal_mask)
        structure_mask=exclude_protected_pixels(structure_mask,all_protected_mask)
        # Expand tight OCR rectangles to include anti-aliased glyph edges, then apply the
        # combined full-resolution mask directly to the original RGB pixels. The previous
        # implementation restored *all dark pixels* after inpainting, which restored the
        # black measurement glyphs inside the mask and made the candidate nearly identical.
        set_stage(jid,'inpainting',72,'Reconstructing masked pixels')
        out=apply_removal_mask(rgb,annotation_mask,dilation=2,protected=all_protected_mask)
        out=apply_removal_mask(out,structure_mask,dilation=0,protected=all_protected_mask)
        out[all_protected_mask>0]=rgb[all_protected_mask>0]
        mask=cv2.dilate(annotation_mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
        mask=cv2.bitwise_or(mask,structure_mask)
        mask[all_protected_mask>0]=0
        set_stage(jid,'verification',80,'Verifying changed pixels and protected regions')
        changed=np.any(out!=rgb,axis=2)
        changed_count=int(np.count_nonzero(changed))
        for item in periodic_guides:
            x1,y1,x2,y2=[int(item[k]) for k in ('x1','y1','x2','y2')]
            x0=max(0,min(x1,x2)-4); y0=max(0,min(y1,y2)-4); x3=min(rgb.shape[1],max(x1,x2)+5); y3=min(rgb.shape[0],max(y1,y2)+5)
            local=np.zeros((y3-y0,x3-x0),np.uint8); cv2.line(local,(x1-x0,y1-y0),(x2-x0,y2-y0),255,5)
            selected=cv2.bitwise_and(local,guide_removal_mask[y0:y3,x0:x3])>0
            item['changed_pixels']=int(np.count_nonzero(changed[y0:y3,x0:x3]&selected))
            # Some rejected/ambiguous guide candidates never receive a mask.
            # Missing evidence keeps them in review instead of crashing or succeeding.
            item['removal_verified']=_removal_has_evidence(item)
        for item in box_symbols:
            if not item.get('mask_pixels'): item['changed_pixels']=0; item['removal_verified']=False; continue
            x0=max(0,int(item['x'])-1); y0=max(0,int(item['y'])-1); x1=min(rgb.shape[1],int(item['x']+item['width']+1)); y1=min(rgb.shape[0],int(item['y']+item['height']+1))
            selected=symbol_removal_mask[y0:y1,x0:x1]>0
            item['changed_pixels']=int(np.count_nonzero(changed[y0:y1,x0:x1]&selected))
            item['removal_verified']=bool(item['mask_pixels'] and item['changed_pixels'])
        def verified_boxes(boxes):
            verified=[]; pending=[]
            for box in boxes:
                x,y,w,h=[int(box.get(k,0)) for k in ('x','y','width','height')]
                x1=min(rgb.shape[1],x+w); y1=min(rgb.shape[0],y+h)
                if x1<=x or y1<=y: pending.append(dict(box,mask_pixels=0,changed_pixels=0,removal_verified=False)); continue
                roi_mask=mask[y:y1,x:x1]; roi_changed=changed[y:y1,x:x1]
                mp=int(cv2.countNonZero(roi_mask)); cp=int(np.count_nonzero(roi_changed))
                annotated=dict(box,mask_pixels=mp,changed_pixels=cp,removal_verified=bool(mp and cp>max(4,int(w*h*.01))))
                (verified if annotated['removal_verified'] else pending).append(annotated)
            return verified,pending
        effective_measurements=accepted+merged_labels
        verified_measurements,unverified_measurements=verified_boxes(effective_measurements)
        verified_red,unverified_red=verified_boxes(red_boxes)
        set_stage(jid,'diagnostics',88,'Saving diagnostic overlays and review artifacts')
        # Persist same-run diagnostics for end-to-end inspection.
        overlay=rgb.copy(); overlay[mask>0]=((overlay[mask>0].astype(np.uint16)+np.array([255,30,30],np.uint16))//2).astype(np.uint8)
        boxes=rgb.copy()
        for b in detected:
            x,y,w,h=[int(b.get(k,0)) for k in ('x','y','width','height')]
            cv2.rectangle(boxes,(x,y),(min(rgb.shape[1]-1,x+w),min(rgb.shape[0]-1,y+h)),(20,90,255),1)
        (PROCESSED_DIR/f'{jid}-original.png').write_bytes(encode_png(rgb))
        (PROCESSED_DIR/f'{jid}-ocr-boxes.png').write_bytes(encode_png(boxes))
        (PROCESSED_DIR/f'{jid}-mask-overlay.png').write_bytes(encode_png(overlay))
        symbol_mask=symbol_removal_mask; guide_mask=guide_removal_mask
        structure_overlay=rgb.copy()
        structure_overlay[guide_mask>0]=(255,120,20)
        structure_overlay[symbol_mask>0]=(20,100,255)
        for b in periodic_guides:
            if b.get('classification')!='high-confidence-removable': cv2.line(structure_overlay,(b['x1'],b['y1']),(b['x2'],b['y2']),(255,220,0),1)
        for b in box_symbols:
            if b.get('classification')!='high-confidence-removable': cv2.rectangle(structure_overlay,(b['x'],b['y']),(b['x']+b['width'],b['y']+b['height']),(255,0,220),1)
        (PROCESSED_DIR/f'{jid}-structure-proposals.png').write_bytes(encode_png(structure_overlay))
        for category,category_mask in (('measurement-mask',measurement_mask),('protected-number-mask',protected_mask),('symbol-mask',symbol_mask),('guide-mask',guide_mask)):
            (PROCESSED_DIR/f'{jid}-{category}.png').write_bytes(encode_png(cv2.cvtColor(category_mask,cv2.COLOR_GRAY2RGB)))
        # Crop the largest confidently recognized decimal for a reproducible original/cleaned check.
        crop_meta=None
        if effective_measurements:
            b=max(effective_measurements,key=lambda q:q['width']*q['height']); pad=max(30,b['height']*2)
            x0=max(0,b['x']-pad); y0=max(0,b['y']-pad); x1=min(rgb.shape[1],b['x']+b['width']+pad); y1=min(rgb.shape[0],b['y']+b['height']+pad)
            original_crop=cv2.resize(rgb[y0:y1,x0:x1],None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
            cleaned_crop=cv2.resize(out[y0:y1,x0:x1],None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(PROCESSED_DIR/f'{jid}-crop-original.png'),cv2.cvtColor(original_crop,cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(PROCESSED_DIR/f'{jid}-crop-cleaned.png'),cv2.cvtColor(cleaned_crop,cv2.COLOR_RGB2BGR))
            side_by_side=np.full((max(original_crop.shape[0],cleaned_crop.shape[0])+36,original_crop.shape[1]+cleaned_crop.shape[1],3),255,np.uint8)
            side_by_side[36:36+original_crop.shape[0],:original_crop.shape[1]]=original_crop
            side_by_side[36:36+cleaned_crop.shape[0],original_crop.shape[1]:]=cleaned_crop
            cv2.putText(side_by_side,'ORIGINAL',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
            cv2.putText(side_by_side,'CLEANED',(original_crop.shape[1]+8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),1)
            cv2.imwrite(str(PROCESSED_DIR/f'{jid}-crop-comparison.png'),cv2.cvtColor(side_by_side,cv2.COLOR_RGB2BGR))
            crop_meta={'text':b['text'],'bbox_original':[b['x'],b['y'],b['width'],b['height']],'crop':[x0,y0,x1,y1],'comparison_url':f'/api/jobs/{jid}/diagnostic/crop-comparison'}
        set_stage(jid,'preview_generation',96,'Generating candidate preview')
        outpath=PROCESSED_DIR/f'{jid}.png'; outpath.write_bytes(encode_png(out))
        # Components and line candidates are review signals, never blanket deletion rules.
        diagnostics={'original_url':f'/api/jobs/{jid}/diagnostic/original','ocr_boxes_url':f'/api/jobs/{jid}/diagnostic/ocr-boxes','mask_overlay_url':f'/api/jobs/{jid}/diagnostic/mask-overlay','measurement_mask_url':f'/api/jobs/{jid}/diagnostic/measurement-mask','protected_number_mask_url':f'/api/jobs/{jid}/diagnostic/protected-number-mask','symbol_mask_url':f'/api/jobs/{jid}/diagnostic/symbol-mask','guide_mask_url':f'/api/jobs/{jid}/diagnostic/guide-mask','structure_proposals_url':f'/api/jobs/{jid}/diagnostic/structure-proposals','crop_original_url':f'/api/jobs/{jid}/diagnostic/crop-original','crop_cleaned_url':f'/api/jobs/{jid}/diagnostic/crop-cleaned','crop_comparison_url':f'/api/jobs/{jid}/diagnostic/crop-comparison','selected_detection':crop_meta}
        removed_guide_count=sum(bool(b.get('removal_verified')) and b.get('classification')=='high-confidence-removable' for b in periodic_guides)
        partial_guides_for_review=sum(bool(b.get('mask_pixels')) and b.get('review_required') for b in periodic_guides)
        removed_symbol_count=sum(bool(b.get('removal_verified')) for b in box_symbols)
        review={'asset_id':aid,'detections':detected,'measurement_candidates':effective_measurements,'removed':verified_measurements,'unverified_measurements':unverified_measurements,'uncertain':uncertain,'protected_principal_labels':[{'text':b.get('text',b.get('protected_text','')),'x':b['x'],'y':b['y'],'width':b['width'],'height':b['height']} for b in unique_principal_boxes],'removed_red_components':len(verified_red),'unverified_red_components':len(unverified_red),'red_remaining_components':len(unverified_red),'unresolved_decimal_like':uncertain+unverified_measurements,'dashed_candidates':dashed,'periodic_line_candidates':periodic_guides,'box_symbol_candidates':box_symbols,'ocr_warnings':ocr_warnings,'mask_url':f'/api/jobs/{jid}/mask','approved':False,'diagnostics':diagnostics,'qc':{'boundary_comparison':'review-required','candidate_line_count':len(dashed),'periodic_line_candidate_count':len(periodic_guides),'high_confidence_guides_removed':removed_guide_count,'ambiguous_guides_for_review':sum(b.get('classification')=='ambiguous-review' for b in periodic_guides),'box_symbol_candidate_count':len(box_symbols),'high_confidence_symbols_removed':removed_symbol_count,'ambiguous_symbols_for_review':sum(b.get('classification')=='ambiguous-review' for b in box_symbols),'symbol_removal_implemented':bool(removed_symbol_count),'guide_removal_implemented':bool(removed_guide_count),'guide_changed_pixels':int(np.count_nonzero(changed&(guide_removal_mask>0))),'symbol_changed_pixels':int(np.count_nonzero(changed&(symbol_removal_mask>0))),'protected_boundary_pixels':int(cv2.countNonZero(boundary_protected_mask)),'ocr_available':bool(detected or red_boxes),'source_dimensions':[int(rgb.shape[1]),int(rgb.shape[0])],'mask_dimensions':[int(mask.shape[1]),int(mask.shape[0])],'mask_nonzero_pixels':int(cv2.countNonZero(mask)),'ocr_detection_count':len(detected),'removal_regions':len(effective_measurements)+len(red_boxes)+removed_guide_count+removed_symbol_count,'verified_measurement_removals':len(verified_measurements),'verified_red_removals':len(verified_red),'protected_principal_labels':len(unique_principal_boxes),'changed_pixels':changed_count}}
        review['qc']['partially_removed_guides_for_review']=partial_guides_for_review
        mpath=PROCESSED_DIR/f'{jid}-mask.png'; mpath.write_bytes(encode_png(cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)))
        (PROCESSED_DIR/f'{jid}.json').write_text(json.dumps(review))
        finish_job_timing(jid)
        with lock:
            if jid in jobs:
                review['qc']['stage_timings']=dict(jobs[jid].get('stage_timings',{}))
                review['qc']['elapsed_seconds']=jobs[jid].get('elapsed_seconds')
                review['qc']['peak_memory_mb']=jobs[jid].get('peak_memory_mb')
        (PROCESSED_DIR/f'{jid}.json').write_text(json.dumps(review))
        update_job(jid,status='completed',stage='completed',progress=100,message=f'Candidate ready; {len(verified_measurements)} measurement regions and {len(verified_red)} red regions verified as changed. Review required.',preview_available=True,review=review,candidate_version=0,candidate_path=str(outpath),manual_operations=[])
    except Exception as e:
        finish_job_timing(jid)
        logger.exception('Cleaning job %s failed for asset %s',jid,aid)
        update_job(jid,status='failed',stage='failed',message='Image processing failed',error=f'{type(e).__name__}: {e}',elapsed_seconds=jobs.get(jid,{}).get('elapsed_seconds'),stage_timings=jobs.get(jid,{}).get('stage_timings',{}))

def clean_candidate(jid, aid, path):
    # Full-resolution OCR/segmentation holds several image-sized arrays. Keep the
    # default to one active cleaner so a batch cannot multiply peak memory use.
    with CLEANING_SLOTS:
        _clean_candidate(jid,aid,path)

@app.get('/api/health')
def health(): return {'status':'ok','app':'Survey Map Studio API'}
@app.get('/api/maps')
def list_maps():
    with lock: return [public_asset(a) for a in assets.values()]
@app.post('/api/maps/upload')
async def upload_map(file:UploadFile=File(...),map_type:Literal['primary','sheet']=Form(...)):
    name=Path(file.filename or 'survey-map.png').name; suffix=Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES: raise HTTPException(415,'Unsupported image format.')
    content=await file.read(MAX_UPLOAD_BYTES+1)
    if not content: raise HTTPException(400,'Uploaded file is empty.')
    if len(content)>MAX_UPLOAD_BYTES: raise HTTPException(413,'File exceeds the 40 MB limit.')
    try:
        with Image.open(io.BytesIO(content)) as im: im.verify()
        with Image.open(io.BytesIO(content)) as im: width,height=im.size
        if width<50 or height<50: raise HTTPException(400,'Image is too small.')
        asset_id=uuid.uuid4().hex; path=ORIGINALS_DIR/f'{asset_id}{suffix}'; path.write_bytes(content)
        make_thumbnail(path,asset_id)
    except HTTPException: raise
    except (UnidentifiedImageError,OSError,ValueError) as e: raise HTTPException(415,'Image is corrupt or unsupported.') from e
    a={'id':asset_id,'filename':name,'map_type':map_type,'width':width,'height':height,'uploaded_at':now_iso(),'source_path':str(path),'original_size':len(content)}
    with lock: assets[asset_id]=a
    persist_assets()
    return public_asset(a)
def get_asset(aid):
    with lock: a=assets.get(aid)
    if not a: raise HTTPException(404,'Map not found.')
    return a

def _references_any(value, identifiers:set[str])->bool:
    if isinstance(value,str): return value in identifiers
    if isinstance(value,dict): return any(str(key) in identifiers or _references_any(item,identifiers) for key,item in value.items())
    if isinstance(value,(list,tuple)): return any(_references_any(item,identifiers) for item in value)
    return False

def _removal_has_evidence(candidate:dict)->bool:
    """Return true only when a candidate has both mask and changed-pixel evidence."""
    try:
        return int(candidate.get('mask_pixels') or 0)>0 and int(candidate.get('changed_pixels') or 0)>0
    except (TypeError,ValueError):
        return False

def _safe_owned_files(directory:Path, job_id:str)->list[Path]:
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',job_id): return []
    directory=directory.resolve(); found=[]
    for path in directory.iterdir():
        if not path.is_file() or not (path.name==job_id or path.name.startswith(job_id+'.') or path.name.startswith(job_id+'-')): continue
        try:
            if path.resolve().parent==directory: found.append(path)
        except OSError: continue
    return found

@app.delete('/api/maps/{asset_id}')
def delete_map(asset_id:str):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',asset_id): raise HTTPException(400,'Invalid map ID.')
    asset=get_asset(asset_id); normalized=normalize_asset_record(asset)
    source_text=normalized.get('source_path')
    source=Path(source_text).resolve() if source_text else None
    if not source:
        matches=[p for p in ORIGINALS_DIR.glob(f'{asset_id}.*') if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES]
        if len(matches)>1: raise HTTPException(409,'Deletion is blocked because multiple originals match this asset ID.')
    if source and (source.parent!=ORIGINALS_DIR.resolve() or source.stem!=asset_id):
        raise HTTPException(409,'Deletion is blocked because this asset does not resolve to its own managed original file.')
    with lock:
        related_jobs={jid:dict(job) for jid,job in jobs.items() if job.get('asset_id')==asset_id}
    related_job_ids=set(related_jobs)
    references={asset_id,*related_job_ids}
    for project_file in PROJECTS_DIR.glob('*.json'):
        try: project_data=json.loads(project_file.read_text())
        except (OSError,ValueError) as exc: raise HTTPException(409,'Deletion is blocked because an assembly project could not be checked safely.') from exc
        if _references_any(project_data,references):
            raise HTTPException(409,f'Deletion is blocked: assembly project "{project_file.stem}" references this map or one of its cleaning jobs.')
    owned=[]
    for jid in related_job_ids:
        owned.extend(_safe_owned_files(PROCESSED_DIR,jid))
        for directory in (JOBS_DIR,MANUAL_EDITS_DIR):
            owned.extend(_safe_owned_files(directory,jid))
    thumbnail_path=THUMB_DIR/f'{asset_id}.png'
    owned.extend(_safe_owned_files(THUMB_DIR,asset_id))
    if source and source.is_file(): owned.append(source)
    # Preflight every path before unlinking so no record can direct deletion outside managed storage.
    approved_roots={p.resolve() for p in (ORIGINALS_DIR,THUMB_DIR,PROCESSED_DIR,JOBS_DIR,MANUAL_EDITS_DIR)}
    for path in owned:
        if path.resolve().parent not in approved_roots:
            raise HTTPException(409,'Deletion is blocked because an associated artifact is outside managed storage.')
    removed=0
    try:
        for path in dict.fromkeys(owned):
            path.unlink(missing_ok=True); removed+=1
    except OSError as exc:
        logger.exception('Could not completely delete map %s',asset_id)
        raise HTTPException(500,'Deletion stopped because a managed file could not be removed. The map remains in the library; inspect its files before retrying.') from exc
    with lock:
        assets.pop(asset_id,None)
        for jid in related_job_ids: jobs.pop(jid,None)
    persist_assets()
    return {'deleted':True,'asset_id':asset_id,'jobs_removed':len(related_job_ids),'files_removed':removed}

@app.get('/api/maps/{asset_id}/image')
def image(asset_id:str):
    a=get_asset(asset_id); source=a.get('source_path')
    if not source: raise HTTPException(404,'Original image is unavailable for this legacy map record.')
    p=Path(source)
    if not p.exists(): raise HTTPException(404,'Original image file is missing.')
    try: return Response(encode_png(read_rgb(p)),media_type='image/png',headers={'X-Original-Format':p.suffix.lower()})
    except Exception as e: raise HTTPException(422,'Could not decode original image.') from e
@app.get('/api/maps/{asset_id}/thumbnail')
def thumbnail(asset_id:str):
    a=get_asset(asset_id); p=THUMB_DIR/f'{asset_id}.png'
    try:
        if p.exists():
            try:
                with Image.open(p) as image: image.verify()
            except (OSError,UnidentifiedImageError,ValueError): p.unlink(missing_ok=True)
        if not p.exists():
            source=a.get('source_path')
            if not source or not Path(source).is_file(): raise FileNotFoundError('Original image is unavailable.')
            make_thumbnail(Path(source),asset_id)
        return Response(p.read_bytes(),media_type='image/png',headers={'Cache-Control':'no-store'})
    except (OSError,UnidentifiedImageError,ValueError) as exc:
        p.unlink(missing_ok=True)
        raise HTTPException(404,'Thumbnail is unavailable and could not be regenerated from the original.') from exc
@app.get('/api/maps/{asset_id}/preview')
def map_preview(asset_id:str):
    a=get_asset(asset_id); jid=a.get('latest_job_id')
    p=current_candidate_path(jid) if jid else None
    if jid and p.exists(): return Response(p.read_bytes(),media_type='image/png',headers={'Cache-Control':'no-store','X-Candidate-Job':jid})
    source=Path(a.get('source_path') or '')
    if not source.is_file(): raise HTTPException(404,'Original image is unavailable for preview.')
    try: return Response(encode_png(read_rgb(source)),media_type='image/png',headers={'Cache-Control':'no-store','X-Preview-Fallback':'original'})
    except Exception as exc: raise HTTPException(422,'Could not render an original-image preview.') from exc
@app.post('/api/maps/{asset_id}/process')
def process_map(asset_id:str,bg:BackgroundTasks):
    a=get_asset(asset_id)
    with lock:
        active=next((j for j in jobs.values() if j.get('asset_id')==asset_id and j.get('status') in {'queued','processing'}),None)
        if active: return {'job_id':active['id'],'status_url':f"/api/jobs/{active['id']}",'reused':True}
        jid=uuid.uuid4().hex; stamp=now_iso()
        jobs[jid]={'id':jid,'asset_id':asset_id,'status':'queued','stage':'queued','progress':0,'message':'Queued','preview_available':False,'created_at':stamp,'updated_at':stamp,'stage_timings':{},'_started_perf':time.perf_counter()}
        a['latest_job_id']=jid; a['approved']=False
    persist_job(jid); persist_assets()
    bg.add_task(clean_candidate,jid,asset_id,Path(a['source_path']))
    return {'job_id':jid,'status_url':f'/api/jobs/{jid}'}
@app.get('/api/jobs/{job_id}')
def get_job(job_id:str):
    with lock: j=jobs.get(job_id)
    if not j: raise HTTPException(404,'Job not found.')
    result={k:v for k,v in j.items() if not k.startswith('_')}
    if j.get('status') in {'queued','processing'}: result['elapsed_seconds']=round(time.perf_counter()-j.get('_started_perf',time.perf_counter()),1)
    result['preview_url']=f'/api/jobs/{job_id}/preview' if j.get('preview_available') else None
    return result
@app.get('/api/jobs/{job_id}/preview')
def job_preview(job_id:str):
    with lock: j=jobs.get(job_id)
    p=current_candidate_path(job_id,j)
    if not j or not j.get('preview_available') or not p.exists(): raise HTTPException(404,'Candidate preview is unavailable.')
    return Response(p.read_bytes(),media_type='image/png',headers={'Cache-Control':'no-store','X-Candidate-Job':job_id})
@app.get('/api/jobs/{job_id}/mask')
def job_mask(job_id:str):
    with lock: j=jobs.get(job_id)
    p=PROCESSED_DIR/f'{job_id}-mask.png'
    if not j or not p.exists(): raise HTTPException(404,'Removal mask is unavailable.')
    return Response(p.read_bytes(),media_type='image/png')
@app.get('/api/jobs/{job_id}/diagnostic/{name}')
def diagnostic(job_id:str,name:str):
    allowed={'original','ocr-boxes','mask-overlay','measurement-mask','protected-number-mask','symbol-mask','guide-mask','structure-proposals','crop-original','crop-cleaned','crop-comparison'}
    if name not in allowed: raise HTTPException(404,'Diagnostic image not found.')
    with lock: j=jobs.get(job_id)
    p=PROCESSED_DIR/f'{job_id}-{name}.png'
    if not j or not p.exists(): raise HTTPException(404,'Diagnostic image not found.')
    return Response(p.read_bytes(),media_type='image/png',headers={'Cache-Control':'no-store'})
@app.get('/api/jobs/{job_id}/manual-edits')
def get_manual_edits(job_id:str):
    with lock: j=jobs.get(job_id)
    if not j or not j.get('preview_available'): raise HTTPException(404,'Candidate not found.')
    return {'job_id':job_id,'candidate_version':j.get('candidate_version',0),'operations':j.get('manual_operations',[]),'source_asset_id':j['asset_id']}
@app.put('/api/jobs/{job_id}/manual-edits')
def save_manual_edits(job_id:str,body:dict):
    with lock:
        j=jobs.get(job_id)
        if not j or j.get('status')!='completed': raise HTTPException(404,'Candidate not found.')
        job_lock=manual_locks.setdefault(job_id,Lock())
    with job_lock:
        with lock:
            j=jobs[job_id]; version=int(j.get('candidate_version',0)); asset=assets.get(j['asset_id'])
            if body.get('candidate_version')!=version: raise HTTPException(409,'Candidate changed since this edit began. Reload edits before saving.')
            width,height=int(asset['width']),int(asset['height'])
        operations=body.get('operations')
        if not isinstance(operations,list) or len(operations)>5000: raise HTTPException(422,'Edits must be a list of at most 5,000 operations.')
        allowed_tools={'brush','background','erase','line','repair','selection-clear'}
        for op in operations:
            if not isinstance(op,dict) or op.get('tool') not in allowed_tools: raise HTTPException(422,'Unsupported edit operation.')
            points=op.get('points',[])
            if not isinstance(points,list) or len(points)>100000: raise HTTPException(422,'An edit has too many points.')
            try:
                for point in points:
                    if len(point)!=2 or not (0<=float(point[0])<width and 0<=float(point[1])<height): raise ValueError
                stroke_width=int(op.get('width',1))
                if not 1<=stroke_width<=4096: raise ValueError
                if op.get('tool')!='erase' and not re.fullmatch(r'#[0-9a-fA-F]{6}',str(op.get('color','#ffffff'))): raise ValueError
                if op.get('tool')=='repair':
                    sx,sy=op['sample']
                    if not (0<=int(sx)<width and 0<=int(sy)<height): raise ValueError
            except (TypeError,ValueError,KeyError): raise HTTPException(422,'Edit coordinates, brush size, or color are invalid for this image.')
        base=PROCESSED_DIR/f'{job_id}.png'
        if not base.exists(): raise HTTPException(404,'Automatic candidate is unavailable.')
        next_version=version+1; target=PROCESSED_DIR/f'{job_id}-manual-v{next_version}.png'
        try:
            if operations: apply_manual_operations(base,target,operations)
            else:
                import shutil
                shutil.copyfile(base,target)
        except Exception as exc:
            logger.exception('Manual edit save failed for job %s',job_id)
            target.unlink(missing_ok=True)
            raise HTTPException(500,'Could not apply these edits. The previous candidate is intact.') from exc
        manifest={'job_id':job_id,'source_asset_id':asset['id'],'candidate_version':next_version,'operations':operations}
        manifest_path=MANUAL_EDITS_DIR/f'{job_id}.json'; temp=manifest_path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest)); temp.replace(manifest_path)
        with lock:
            j=jobs[job_id]; j.update(candidate_version=next_version,candidate_path=str(target),manual_operations=operations,approved=False,updated_at=now_iso(),message='Manual edits saved; approval cleared.')
            asset['approved']=False
        persist_job(job_id); persist_assets()
        return {'saved':True,'approved':False,'candidate_version':next_version,'preview_url':f'/api/jobs/{job_id}/preview?v={next_version}'}
@app.post('/api/jobs/{job_id}/edit')
def edit_candidate(job_id:str,body:dict):
    with lock:
        j=jobs.get(job_id)
        if not j or not j.get('preview_available'): raise HTTPException(404,'Candidate not found.')
        j['approved']=False; assets[j['asset_id']]['approved']=False
        j['updated_at']=now_iso(); j['message']='Candidate changed; review approval cleared.'
    persist_job(job_id); persist_assets()
    return {'updated':True,'approved':False}
class Approval(BaseModel): approved:bool
@app.post('/api/jobs/{job_id}/approve')
def approve(job_id:str,body:Approval):
    with lock:
        j=jobs.get(job_id)
        if not j or j.get('status')!='completed': raise HTTPException(409,'Candidate is not ready for review.')
        if not body.approved: raise HTTPException(400,'Explicit approval is required.')
        assets[j['asset_id']]['approved']=True; j['approved']=True; j['approved_version']=j.get('candidate_version',0); j['message']='Cleaned sheet approved'
    persist_job(job_id); persist_assets()
    return {'approved':True,'download_url':f'/api/jobs/{job_id}/export.png'}
@app.get('/api/jobs/{job_id}/export.png')
def export_png(job_id:str):
    with lock: j=jobs.get(job_id)
    if not j or not j.get('approved') or j.get('approved_version')!=j.get('candidate_version',0): raise HTTPException(403,'Approve the current candidate version before downloading.')
    p=current_candidate_path(job_id,j)
    if not p.exists(): raise HTTPException(404,'Approved candidate artifact is missing.')
    return Response(p.read_bytes(),media_type='image/png',headers={'Content-Disposition':'attachment; filename=cleaned-survey-sheet.png','X-Candidate-Version':str(j.get('candidate_version',0))})
@app.get('/api/jobs/{job_id}/export.pdf')
def export_pdf(job_id:str):
    with lock: j=jobs.get(job_id)
    if not j or not j.get('approved') or j.get('approved_version')!=j.get('candidate_version',0): raise HTTPException(403,'Approve the current candidate version before downloading.')
    p=current_candidate_path(job_id,j)
    if not p.exists(): raise HTTPException(404,'Approved candidate artifact is missing.')
    b=io.BytesIO(); Image.open(p).convert('RGB').save(b,format='PDF',resolution=300)
    return Response(b.getvalue(),media_type='application/pdf',headers={'Content-Disposition':'attachment; filename=cleaned-survey-sheet.pdf'})
class ProjectBody(BaseModel): layout:dict
@app.put('/api/projects/{project_id}')
def save_project(project_id:str,body:ProjectBody):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',project_id): raise HTTPException(400,'Invalid project id.')
    projects[project_id]=body.layout; (PROJECTS_DIR/f'{project_id}.json').write_text(json.dumps(body.layout)); return {'id':project_id,'saved':True}
@app.get('/api/projects/{project_id}')
def load_project(project_id:str):
    p=PROJECTS_DIR/f'{project_id}.json'
    if not p.exists(): raise HTTPException(404,'Assembly project not found.')
    return json.loads(p.read_text())

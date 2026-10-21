"""Map-agnostic structure detection with pixel-local conservative removal masks."""
from __future__ import annotations

import cv2
import numpy as np


def detect_box_symbols(rgb: np.ndarray, tile_size: int = 1024) -> list[dict]:
    h, w = rgb.shape[:2]; found=[]; seen=set(); overlap=48; step=tile_size-overlap
    for y0 in range(0,h,step):
        for x0 in range(0,w,step):
            tile=rgb[y0:min(h,y0+tile_size),x0:min(w,x0+tile_size)]
            gray=cv2.cvtColor(tile,cv2.COLOR_RGB2GRAY)
            binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
            contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area=cv2.contourArea(contour)
                if area<30 or area>tile.shape[0]*tile.shape[1]*.12: continue
                perimeter=cv2.arcLength(contour,True)
                if perimeter<=0: continue
                poly=cv2.approxPolyDP(contour,.035*perimeter,True)
                if not cv2.isContourConvex(poly) or not 4<=len(poly)<=8: continue
                x,y,bw,bh=cv2.boundingRect(poly)
                if min(bw,bh)<8 or max(bw,bh)>max(tile.shape[:2])*.45: continue
                aspect=bw/max(1,bh)
                if not .2<=aspect<=5: continue
                rectangularity=area/max(1,bw*bh)
                if rectangularity<.58: continue
                roi=tile[y:min(tile.shape[0],y+bh),x:min(tile.shape[1],x+bw)]
                hsv=cv2.cvtColor(roi,cv2.COLOR_RGB2HSV)
                blue=cv2.inRange(hsv,np.array([82,38,32],np.uint8),np.array([153,255,255],np.uint8))
                blue_pixels=int(cv2.countNonZero(blue))
                component_count=int(cv2.connectedComponents((blue>0).astype(np.uint8),8)[0]-1) if blue_pixels else 0
                # Inner grayscale contours alone do not prove a closed schematic box.
                # The external blue contour pass below checks all four border edges.
                hatch_confident=False
                gx,gy=x0+x,y0+y; key=(round(gx/4),round(gy/4),round(bw/4),round(bh/4))
                if key in seen: continue
                seen.add(key)
                found.append({'x':gx,'y':gy,'width':bw,'height':bh,'kind':'blue-hatched-symbol' if hatch_confident else 'box-shaped-symbol-review','classification':'high-confidence-removable' if hatch_confident else 'ambiguous-review','confidence':round(min(1,rectangularity),3),'blue_pixels':blue_pixels,'blue_components':component_count})
            hsv_tile=cv2.cvtColor(tile,cv2.COLOR_RGB2HSV)
            blue_tile=cv2.inRange(hsv_tile,np.array([82,38,32],np.uint8),np.array([153,255,255],np.uint8))
            joined_blue=cv2.morphologyEx(blue_tile,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            joined_blue=cv2.dilate(joined_blue,np.ones((3,3),np.uint8),iterations=1)
            blue_contours,_=cv2.findContours(joined_blue,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in blue_contours:
                area=cv2.contourArea(contour)
                if area<30: continue
                x,y,bw,bh=cv2.boundingRect(contour)
                if min(bw,bh)<14 or max(bw,bh)>max(tile.shape[:2])*.45: continue
                aspect=bw/max(1,bh)
                if not .2<=aspect<=5: continue
                roi=joined_blue[y:y+bh,x:x+bw]
                edge=max(1,min(bw,bh)//12)
                edge_coverage=(float(np.count_nonzero(roi[:edge,:]))/(edge*bw),float(np.count_nonzero(roi[-edge:,:]))/(edge*bw),float(np.count_nonzero(roi[:,:edge]))/(edge*bh),float(np.count_nonzero(roi[:,-edge:]))/(edge*bh))
                interior=roi[edge:-edge,edge:-edge] if min(roi.shape)>2*edge else roi
                hatch_rows=int(np.count_nonzero((interior>0).sum(axis=1)>=max(4,int(interior.shape[1]*.35))))
                row_runs=int(np.count_nonzero((interior>0).sum(axis=1)>=max(4,int(interior.shape[1]*.45))))
                col_runs=int(np.count_nonzero((interior>0).sum(axis=0)>=max(4,int(interior.shape[0]*.45))))
                edge_hits=sum(value>=.28 for value in edge_coverage)
                hatch_confident=edge_hits>=3 and (row_runs>=2 or col_runs>=2 or hatch_rows>=3)
                gx,gy=x0+x,y0+y; key=(round(gx/4),round(gy/4),round(bw/4),round(bh/4))
                if key in seen:
                    previous=next((b for b in found if (round(b['x']/4),round(b['y']/4),round(b['width']/4),round(b['height']/4))==key),None)
                    if previous is not None and hatch_confident:
                        previous.update(kind='blue-hatched-symbol',classification='high-confidence-removable',confidence=round(min(edge_coverage),3),blue_pixels=int(cv2.countNonZero(roi)),blue_components=hatch_rows,hatch_rows=hatch_rows)
                    continue
                seen.add(key)
                found.append({'x':gx,'y':gy,'width':bw,'height':bh,'kind':'blue-hatched-symbol' if hatch_confident else 'blue-box-review','classification':'high-confidence-removable' if hatch_confident else 'ambiguous-review','confidence':round(min(1.0,edge_hits/4+(row_runs+col_runs)*.05) if hatch_confident else max(edge_coverage),3),'blue_pixels':int(cv2.countNonZero(blue_tile[y:y+bh,x:x+bw])),'blue_components':hatch_rows,'hatch_rows':hatch_rows,'internal_row_runs':row_runs,'internal_column_runs':col_runs})
    merged=[]
    for candidate in sorted(found,key=lambda b:(b['x'],b['y'])):
        joined=False
        if candidate.get('classification')=='high-confidence-removable':
            for current in merged:
                if current.get('classification')!='high-confidence-removable': continue
                left=max(current['x'],candidate['x']); right=min(current['x']+current['width'],candidate['x']+candidate['width'])
                overlap=(right-left)/max(1,min(current['width'],candidate['width']))
                gap=max(0,max(current['y'],candidate['y'])-min(current['y']+current['height'],candidate['y']+candidate['height']))
                if overlap>=.8 and gap<=max(4,min(current['height'],candidate['height'])*.25):
                    x0=min(current['x'],candidate['x']); y0=min(current['y'],candidate['y'])
                    x1=max(current['x']+current['width'],candidate['x']+candidate['width']); y1=max(current['y']+current['height'],candidate['y']+candidate['height'])
                    current.update(x=x0,y=y0,width=x1-x0,height=y1-y0,blue_pixels=current['blue_pixels']+candidate['blue_pixels'],blue_components=current.get('blue_components',0)+candidate.get('blue_components',0),hatch_rows=current.get('hatch_rows',0)+candidate.get('hatch_rows',0))
                    joined=True; break
        if not joined: merged.append(dict(candidate))
    return merged


def _component_dash_trains(gray: np.ndarray, rgb: np.ndarray | None = None,
                          tile_size: int = 1024) -> list[dict]:
    """Find repeated, detached stroke components and retain their exact boxes.

    Unlike Hough corridors, these proposals can later mask only observed ink
    components. A candidate needs several similarly oriented, slender marks
    with repeat spacing; individual dotted marks are attached to a confirmed
    train only when they lie on its centerline.
    """
    h, w = gray.shape[:2]
    overlap = 48
    step = max(64, tile_size - overlap)
    features: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(h, y0 + tile_size), min(w, x0 + tile_size)
            tile = gray[y0:y1, x0:x1]
            if rgb is None:
                ink = (tile < 165).astype(np.uint8)
            else:
                color = rgb[y0:y1, x0:x1].astype(np.int16)
                chroma = color.max(axis=2) - color.min(axis=2)
                ink = ((tile < 165) & (chroma < 38)).astype(np.uint8)
            count, labels, stats, centers = cv2.connectedComponentsWithStats(ink, 8)
            for label in range(1, count):
                x, y, cw, ch, area = map(int, stats[label])
                # Short dash strokes are common in scanned sheets. Keep compact
                # elongated components as evidence; removal still requires a
                # multi-component periodic train below.
                if area < 8 or max(cw, ch) < 8:
                    continue
                cx, cy = x + cw / 2, y + ch / 2
                # The tile overlap makes the component complete in the owning tile.
                if ((x0 and cx < overlap / 2) or (y0 and cy < overlap / 2) or
                    (x1 < w and cx >= (x1 - x0) - overlap / 2) or
                    (y1 < h and cy >= (y1 - y0) - overlap / 2)):
                    continue
                key = (x0 + x, y0 + y, cw, ch)
                if key in seen:
                    continue
                seen.add(key)
                yy, xx = np.where(labels[y:y + ch, x:x + cw] == label)
                points = np.column_stack((xx + x, yy + y)).astype(np.float32)
                if len(points) < 8:
                    continue
                vx, vy, mx, my = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).flatten()[:4]
                if vx < 0:
                    vx, vy = -vx, -vy
                along = (points[:, 0] - mx) * vx + (points[:, 1] - my) * vy
                normal = (points[:, 0] - mx) * -vy + (points[:, 1] - my) * vx
                length = float(along.max() - along.min())
                thickness = float(normal.max() - normal.min() + 1)
                slenderness = length / max(1.0, thickness)
                # A boundary intersection/branch usually makes the component
                # wider or less line-like; retain those as review-only evidence.
                if length < 8 or thickness > 14 or slenderness < 3.5:
                    continue
                angle = float(np.degrees(np.arctan2(vy, vx)))
                normal_at_origin = float(-vy * (mx + x0) + vx * (my + y0))
                features.append({
                    'box': {'x': x0 + x, 'y': y0 + y, 'width': cw, 'height': ch,
                            'component_area': area, 'component_cx': float(x0 + centers[label][0]),
                            'component_cy': float(y0 + centers[label][1])},
                    'center': (float(mx + x0), float(my + y0)), 'vx': float(vx), 'vy': float(vy),
                    'angle': angle, 'normal': normal_at_origin,
                    'start': float(along.min() + (mx + x0) * vx + (my + y0) * vy),
                    'end': float(along.max() + (mx + x0) * vx + (my + y0) * vy),
                    'length': length, 'thickness': thickness, 'area': area,
                })

    # Build collinear groups. The threshold is relative to stroke thickness;
    # no map dimensions, coordinates or text values are used.
    groups: list[list[dict]] = []
    for feature in sorted(features, key=lambda q: q['angle']):
        matches = []
        for group in groups:
            ref = group[0]
            if abs(feature['angle'] - ref['angle']) > 4.5:
                continue
            if abs(feature['normal'] - ref['normal']) > max(4.0, min(12.0, (feature['thickness'] + ref['thickness']) * .55)):
                continue
            matches.append(group)
        if not matches:
            groups.append([feature])
        else:
            base = matches[0]
            base.append(feature)
            for other in matches[1:]:
                base.extend(other)
                groups.remove(other)

    candidates: list[dict] = []
    for group in groups:
        group.sort(key=lambda q: (q['start'] + q['end']) / 2)
        if len(group) < 4:
            continue
        lengths = np.asarray([q['length'] for q in group], np.float32)
        gaps = np.asarray([group[i + 1]['start'] - group[i]['end'] for i in range(len(group) - 1)], np.float32)
        median_length = float(np.median(lengths))
        # A short aligned word can also form a row of components. A guide train
        # must span several dash lengths before it can become a removal proposal.
        if group[-1]['end'] - group[0]['start'] < median_length * 4.0:
            continue
        valid_gaps = gaps[(gaps >= max(3.0, median_length * .12)) & (gaps <= median_length * 4.0)]
        if len(valid_gaps) < 3:
            continue
        gap_cv = float(np.std(valid_gaps) / (np.mean(valid_gaps) + 1e-6))
        length_cv = float(np.std(lengths) / (np.mean(lengths) + 1e-6))
        if gap_cv > .85 or length_cv > .65:
            continue
        review_required = bool(np.max(gaps) > median_length * 2.5)
        vx = float(np.median([q['vx'] for q in group])); vy = float(np.median([q['vy'] for q in group]))
        first = min(q['start'] for q in group); last = max(q['end'] for q in group)
        # Add only small dot components aligned on this confirmed chain.
        normal_ref = float(np.median([q['normal'] for q in group]))
        p1 = (first * vx - normal_ref * vy, first * vy + normal_ref * vx)
        p2 = (last * vx - normal_ref * vy, last * vy + normal_ref * vx)
        small: list[dict] = []
        pad = 10
        left = max(0, int(min(p1[0], p2[0]) - pad)); right = min(w, int(max(p1[0], p2[0]) + pad + 1))
        top = max(0, int(min(p1[1], p2[1]) - pad)); bottom = min(h, int(max(p1[1], p2[1]) + pad + 1))
        if rgb is None:
            local_ink = (gray[top:bottom, left:right] < 165).astype(np.uint8)
        else:
            color = rgb[top:bottom, left:right].astype(np.int16)
            local_ink = ((gray[top:bottom, left:right] < 165) & ((color.max(axis=2) - color.min(axis=2)) < 38)).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(local_ink, 8)
        for label in range(1, count):
            x, y, cw, ch, area = map(int, stats[label])
            if area < 2 or max(cw, ch) > max(6, median_length * .06):
                continue
            cx, cy = left + centers[label][0], top + centers[label][1]
            along = cx * vx + cy * vy
            normal = -vy * cx + vx * cy
            if first <= along <= last and abs(normal - normal_ref) <= 5.0:
                small.append({'x': left + x, 'y': top + y, 'width': cw, 'height': ch,
                              'component_area': area, 'component_cx': float(left + centers[label][0]),
                              'component_cy': float(top + centers[label][1])})
        boxes = [q['box'] for q in group] + small
        dedup = {(b['x'], b['y'], b['width'], b['height']): b for b in boxes}
        boxes = list(dedup.values())
        candidates.append({
            'x1': int(round(p1[0])), 'y1': int(round(p1[1])),
            'x2': int(round(p2[0])), 'y2': int(round(p2[1])),
            'kind': 'connected-component-dash-train',
            'classification': 'ambiguous-review' if review_required else 'high-confidence-removable',
            'confidence': round(max(0.0, 1.0 - gap_cv / 1.7), 3),
            'dash_runs': len(group), 'dash_gap_cv': round(gap_cv, 3),
            'orientation': 'horizontal' if abs(vy) < .08 else 'vertical' if abs(vx) < .08 else 'diagonal',
            'detection_method': 'collinear-connected-components', 'component_boxes': boxes,
            'review_required': review_required,
            'review_reason': 'long gaps may conceal a boundary or label intersection' if review_required else None,
        })
    return candidates


def detect_dashed_guides(gray: np.ndarray, tile_size: int = 1024,
                         rgb: np.ndarray | None = None) -> list[dict]:
    h,w=gray.shape[:2]; found=[]; seen=set(); overlap=48; step=tile_size-overlap
    for y0 in range(0,h,step):
        for x0 in range(0,w,step):
            tile=gray[y0:min(h,y0+tile_size),x0:min(w,x0+tile_size)]
            # Projection profiles catch horizontal and vertical dash trains directly.
            ink=tile<165
            for axis in (0,1):
                profile=ink.any(axis=axis)
                changes=np.diff(np.r_[False,profile,False].astype(np.int8)); starts=np.flatnonzero(changes==1); ends=np.flatnonzero(changes==-1)
                runs=[(int(a),int(b)) for a,b in zip(starts,ends) if b-a>=2]
                for first in range(len(runs)):
                    cluster=[runs[first]]
                    for current in runs[first+1:]:
                        gap=current[0]-cluster[-1][1]
                        if gap>48: break
                        cluster.append(current)
                        if len(cluster)>=3:
                            gaps=np.diff([q[0] for q in cluster]); cv=float(np.std(gaps)/(np.mean(gaps)+1e-6))
                            if cv<.65 and cluster[-1][1]-cluster[0][0]>=36:
                                start,end=cluster[0][0],cluster[-1][1]-1
                                gaps_between=np.array([cluster[k+1][0]-cluster[k][1] for k in range(len(cluster)-1)],dtype=np.float32)
                                gap_cv=float(np.std(gaps_between)/(np.mean(gaps_between)+1e-6)) if len(gaps_between) else 1.0
                                if axis==0:
                                    row_density=ink[:,start:end].sum(axis=1); line_y=int(np.argmax(row_density))
                                    candidate={'x1':x0+start,'y1':y0+line_y,'x2':x0+end,'y2':y0+line_y}
                                else:
                                    col_density=ink[start:end,:].sum(axis=0); line_x=int(np.argmax(col_density))
                                    candidate={'x1':x0+line_x,'y1':y0+start,'x2':x0+line_x,'y2':y0+end}
                                key=(candidate['x1']//4,candidate['y1']//4,candidate['x2']//4,candidate['y2']//4)
                                if key not in seen:
                                    seen.add(key); confident=(len(cluster)>=4 and gap_cv<=.45) or (len(cluster)==3 and gap_cv<=.18 and end-start>=72)
                                    found.append(candidate|{'kind':'periodic-line-review','classification':'high-confidence-removable' if confident else 'ambiguous-review','confidence':round(max(0,1-gap_cv),3),'dash_runs':len(cluster),'dash_gap_cv':round(gap_cv,3),'orientation':'horizontal' if axis==0 else 'vertical'})
                            break
            edges=cv2.Canny(tile,55,145)
            # Retain individual short strokes as primitives; sparse guides can only
            # be classified after aligned fragments are assembled into a dash train.
            lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=8,minLineLength=8,maxLineGap=5)
            if lines is None: continue
            for row in lines[:,0,:]:
                x1,y1,x2,y2=map(int,row); length=float(np.hypot(x2-x1,y2-y1))
                if length<7: continue
                ux=(x2-x1)/max(length,1); uy=(y2-y1)/max(length,1); nx=-uy; ny=ux
                radius=max(24,min(96,int(length*2)))
                along=np.arange(-radius,length+radius+1,1,dtype=np.float32)
                offsets=np.arange(-2,3,dtype=np.float32)
                sx=np.rint(x1+ux*along[:,None]+nx*offsets[None,:]).astype(np.int32)
                sy=np.rint(y1+uy*along[:,None]+ny*offsets[None,:]).astype(np.int32)
                inside=(sx>=0)&(sx<tile.shape[1])&(sy>=0)&(sy<tile.shape[0])
                support=np.zeros(sx.shape,bool); support[inside]=tile[sy[inside],sx[inside]]<175
                train_ink=support.any(axis=1)
                transitions=np.diff(np.r_[False,train_ink,False].astype(np.int8))
                if np.count_nonzero(transitions==1)<2: continue
                gx,gy=x0+min(x1,x2),y0+min(y1,y2); bw=abs(x2-x1)+1; bh=abs(y2-y1)+1
                key=(round(gx/4),round(gy/4),round(bw/4),round(bh/4))
                if key in seen: continue
                seen.add(key)
                orientation='horizontal' if abs(y2-y1)<=max(2,length*.04) else 'vertical' if abs(x2-x1)<=max(2,length*.04) else 'diagonal'
                found.append({'x1':x0+x1,'y1':y0+y1,'x2':x0+x2,'y2':y0+y2,'kind':'line-fragment','classification':'ambiguous-review','confidence':0.0,'dash_runs':1,'dash_gap_cv':1.0,'orientation':orientation,'detection_method':'hough'})
    hough=[q for q in found if q.get('detection_method')=='hough']; projections=[q for q in found if q.get('detection_method')!='hough']
    chains=[]
    for line in sorted(hough,key=lambda q:min(q['x1'],q['x2'])):
        x1,y1,x2,y2=[int(line[k]) for k in ('x1','y1','x2','y2')]
        if x2<x1: x1,y1,x2,y2=x2,y2,x1,y1
        vx=x2-x1; vy=y2-y1; length=max(1.0,float(np.hypot(vx,vy))); ux=vx/length; uy=vy/length
        placed=False
        for chain in chains:
            angle=abs(ux*chain['ux']+uy*chain['uy'])
            cross=abs((x1-chain['x1'])*chain['uy']-(y1-chain['y1'])*chain['ux'])
            end_projection=(x1-chain['x1'])*chain['ux']+(y1-chain['y1'])*chain['uy']
            gap=max(0.0,end_projection-chain['length'])
            # Guides often have wide gaps and are broken into short Hough strokes.
            # Permit a wider join window, but validate the assembled support as a
            # repeated dash train below before any pixels can be removed.
            if angle>=.99 and cross<=5.0 and gap<=96:
                if end_projection+length>chain['length']:
                    chain['x2']=x2; chain['y2']=y2; chain['length']=end_projection+length
                chain['members'].append(line); placed=True; break
        if not placed: chains.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'ux':ux,'uy':uy,'length':length,'members':[line]})
    for chain in chains:
        x1,y1,x2,y2=[int(round(chain[k])) for k in ('x1','y1','x2','y2')]
        length=float(np.hypot(x2-x1,y2-y1)); ux=(x2-x1)/max(length,1); uy=(y2-y1)/max(length,1); nx=-uy; ny=ux
        t=np.arange(int(length),dtype=np.float32); offsets=np.arange(-3,4,dtype=np.float32)
        xs=np.rint(x1+ux*t[:,None]+nx*offsets[None,:]).astype(np.int32); ys=np.rint(y1+uy*t[:,None]+ny*offsets[None,:]).astype(np.int32)
        inside=(xs>=0)&(xs<gray.shape[1])&(ys>=0)&(ys<gray.shape[0]); values=np.zeros(xs.shape,bool); values[inside]=gray[ys[inside],xs[inside]]<175
        ink=values.any(axis=1); changes=np.diff(np.r_[False,ink,False].astype(np.int8)); starts=np.flatnonzero(changes==1); ends=np.flatnonzero(changes==-1)
        if len(starts)<3: continue
        dash_lengths=ends-starts; gaps=starts[1:]-ends[:-1]; gap_cv=float(np.std(gaps)/(np.mean(gaps)+1e-6)) if len(gaps) else 1.0
        confident=((len(starts)>=4 and gap_cv<=.55) or (len(starts)==3 and gap_cv<=.18 and length>=72)) and np.median(gaps)>=2 and np.median(dash_lengths)<=max(18,np.median(gaps)*1.5)
        projections.append({'x1':x1,'y1':y1,'x2':x2,'y2':y2,'kind':'periodic-line-review','classification':'high-confidence-removable' if confident else 'ambiguous-review','confidence':round(max(0,1-gap_cv),3),'dash_runs':int(len(starts)),'dash_gap_cv':round(gap_cv,3),'orientation':'horizontal' if abs(y2-y1)<=max(2,length*.04) else 'vertical' if abs(x2-x1)<=max(2,length*.04) else 'diagonal','detection_method':'aligned-hough-chain'})
    component_candidates = _component_dash_trains(gray, rgb=rgb, tile_size=tile_size)
    # Avoid duplicate review records when an older detector already found the
    # same path. The component record still wins because it carries exact ink boxes.
    unique_components = []
    for candidate in component_candidates:
        duplicate = False
        for old in projections:
            if old.get('classification') != 'high-confidence-removable':
                continue
            if abs(old['x1'] - candidate['x1']) <= 12 and abs(old['y1'] - candidate['y1']) <= 12:
                duplicate = True
                break
        if not duplicate:
            unique_components.append(candidate)
    return projections + unique_components


def build_guide_removal_mask(gray: np.ndarray, candidates: list[dict], rgb:np.ndarray|None=None,
                             protected_labels: np.ndarray|None=None) -> tuple[np.ndarray,np.ndarray,list[dict]]:
    """Select only dark pixels on high-confidence dash trains; protect crossing solid edges.

    No dilation is applied to the deletion mask. At each guide position, a long dark
    cross-section perpendicular to the guide is treated as a real boundary and restored.
    """
    h,w=gray.shape[:2]; removal=np.zeros((h,w),np.uint8); protected=np.zeros_like(removal); classified=[]
    if rgb is None: boundary_ink=gray<180
    else:
        channels=rgb.astype(np.int16); chroma=channels.max(axis=2)-channels.min(axis=2)
        boundary_ink=(gray<180)&(chroma<38)
    for item in candidates:
        candidate=dict(item)
        if item.get('detection_method') == 'collinear-connected-components':
            # These candidates carry the observed stroke components. Select only
            # their actual dark pixels; never rasterize the intervening guide path.
            selected = np.zeros_like(removal)
            for box in item.get('component_boxes', []):
                bx0=int(box['x']); by0=int(box['y'])
                bx1=bx0+int(box['width']); by1=by0+int(box['height'])
                x0=max(0,bx0-2); y0=max(0,by0-2)
                x1=min(w,bx1+2); y1=min(h,by1+2)
                if x1<=x0 or y1<=y0:
                    continue
                if rgb is None:
                    ink = (gray[y0:y1,x0:x1] < 165).astype(np.uint8)
                else:
                    local_color = rgb[y0:y1,x0:x1].astype(np.int16)
                    local_chroma = local_color.max(axis=2) - local_color.min(axis=2)
                    ink = ((gray[y0:y1,x0:x1] < 165) & (local_chroma < 38)).astype(np.uint8)
                count, labels, stats, centers = cv2.connectedComponentsWithStats(ink, 8)
                expected_area = int(box.get('component_area', 0))
                expected_center = np.array([box.get('component_cx', x0 + (x1-x0)/2) - x0,
                                            box.get('component_cy', y0 + (y1-y0)/2) - y0], np.float32)
                matches = []
                for label in range(1, count):
                    area = int(stats[label, cv2.CC_STAT_AREA])
                    center = centers[label]
                    area_error = abs(area - expected_area) / max(1, expected_area)
                    center_error = float(np.linalg.norm(center - expected_center))
                    if area_error <= .12 and center_error <= 2.5:
                        matches.append((area_error + center_error * .01, label))
                if not matches:
                    continue
                selected_label = min(matches)[1]
                component_pixels = labels == selected_label
                if cv2.countNonZero(component_pixels.astype(np.uint8)) < 2:
                    continue
                if rgb is None:
                    antialias = gray[y0:y1,x0:x1] < 245
                    neutral = np.ones(antialias.shape, bool)
                else:
                    color = rgb[y0:y1,x0:x1].astype(np.int16)
                    neutral = (color.max(axis=2) - color.min(axis=2)) < 38
                    antialias = (gray[y0:y1,x0:x1] < 245) & neutral
                # Cover the anti-aliased edge of this exact component, while
                # carving out neighboring dark components (especially parcel
                # edges) before inpainting.
                dilated = cv2.dilate(component_pixels.astype(np.uint8), np.ones((3,3),np.uint8), iterations=1) > 0
                other_dark = (ink > 0) & ~component_pixels
                neighbor_protection = cv2.dilate(other_dark.astype(np.uint8), np.ones((3,3),np.uint8), iterations=1) > 0
                local_mask = dilated & antialias & ~neighbor_protection
                selected[y0:y1,x0:x1][local_mask] = 255
            if protected_labels is not None:
                selected[protected_labels > 0] = 0
            if rgb is not None and cv2.countNonZero(selected):
                # Telea reads a neighborhood around a mask. Preserve nearby
                # colored labels/annotations pixel-for-pixel instead of letting
                # a black guide's inpainting halo alter them.
                halo = cv2.dilate(selected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
                color = rgb.astype(np.int16)
                chroma = color.max(axis=2) - color.min(axis=2)
                protected[(halo > 0) & (chroma >= 38)] = 255
            candidate['mask_pixels'] = int(cv2.countNonZero(selected))
            candidate['protected_boundary_pixels'] = 0
            candidate['review_required'] = bool(candidate.get('review_required') or not candidate['mask_pixels'])
            candidate['review_reason'] = candidate.get('review_reason') or (None if candidate['mask_pixels'] else 'no isolated dash pixels available')
            candidate['classification'] = ('ambiguous-review' if candidate['review_required'] or not candidate['mask_pixels']
                                           else 'high-confidence-removable')
            removal = cv2.bitwise_or(removal, selected)
            classified.append(candidate)
            continue
        x1,y1,x2,y2=[int(item[k]) for k in ('x1','y1','x2','y2')]
        length=float(np.hypot(x2-x1,y2-y1)); runs=int(item.get('dash_runs',0)); gap_cv=float(item.get('dash_gap_cv',1.0))
        eligible=item.get('classification')=='high-confidence-removable' and runs>=3 and gap_cv<=.55 and length>=24
        if not eligible:
            candidate['classification']='ambiguous-review'; classified.append(candidate); continue
        stroke=max(1,min(4,int(round(min(h,w)*.00025))))
        nx=-(y2-y1)/max(length,1); ny=(x2-x1)/max(length,1)
        boundary_reach=max(12,min(32,int(round(min(h,w)*.01))))
        # Work in a candidate-local ROI. The previous implementation allocated
        # and dilated full-map masks for every proposal; hundreds of candidates
        # caused avoidable full-resolution memory traffic.
        pad=boundary_reach+5
        rx0=max(0,min(x1,x2)-pad); ry0=max(0,min(y1,y2)-pad)
        rx1=min(w,max(x1,x2)+pad+1); ry1=min(h,max(y1,y2)+pad+1)
        if rx1<=rx0 or ry1<=ry0:
            candidate.update(classification='ambiguous-review',mask_pixels=0,review_required=True,
                             review_reason='candidate lies outside the image')
            classified.append(candidate); continue
        local_boundary=boundary_ink[ry0:ry1,rx0:rx1]
        t=np.arange(0,length+1,max(1,stroke),dtype=np.float32)
        offsets=np.arange(-boundary_reach,boundary_reach+1,dtype=np.float32)
        xs=np.rint(x1+(x2-x1)*t[:,None]/max(length,1)+nx*offsets[None,:]).astype(np.int32)
        ys=np.rint(y1+(y2-y1)*t[:,None]/max(length,1)+ny*offsets[None,:]).astype(np.int32)
        inside=(xs>=0)&(xs<w)&(ys>=0)&(ys<h)
        values=np.zeros(xs.shape,bool); values[inside]=boundary_ink[ys[inside],xs[inside]]
        # Derive dash intervals from the center-line neighborhood. The previous
        # corridor mask erased every dark pixel between detected dashes too, which
        # could weaken a real parcel edge running parallel to a guide.
        center_values=values[:,np.abs(offsets)<=stroke+2]
        dash_ink=center_values.any(axis=1)
        transitions=np.diff(np.r_[False,dash_ink,False].astype(np.int8))
        dash_starts=np.flatnonzero(transitions==1); dash_ends=np.flatnonzero(transitions==-1)
        observed_lengths=dash_ends-dash_starts
        observed_gaps=dash_starts[1:]-dash_ends[:-1]
        observed_gap_cv=float(np.std(observed_gaps)/(np.mean(observed_gaps)+1e-6)) if len(observed_gaps) else 1.0
        occupancy=float(np.count_nonzero(dash_ink))/max(1,len(dash_ink))
        # A solid parcel edge interrupted by labels can produce a few apparent
        # runs. Require the sampled line itself to show real blank intervals and
        # periodic spacing before accepting it as a guide.
        train_valid=(len(dash_starts)>=3 and observed_gap_cv<=.45 and
                     np.median(observed_gaps)>=2 and
                     np.median(observed_lengths)<=max(18,np.median(observed_gaps)*1.5) and
                     occupancy<=.65)
        if not train_valid:
            candidate.update(classification='ambiguous-review',mask_pixels=0,protected_boundary_pixels=0,
                             review_required=True,review_reason='observed line support is not a sufficiently sparse periodic dash train')
            classified.append(candidate); continue
        dash_mask=np.zeros((ry1-ry0,rx1-rx0),np.uint8)
        for start,end in zip(dash_starts,dash_ends):
            if end-start<2: continue
            ax=int(round(x1+(x2-x1)*start/max(length,1))); ay=int(round(y1+(y2-y1)*start/max(length,1)))
            bx=int(round(x1+(x2-x1)*(end-1)/max(length,1))); by=int(round(y1+(y2-y1)*(end-1)/max(length,1)))
            cv2.line(dash_mask,(ax-rx0,ay-ry0),(bx-rx0,by-ry0),255,2*stroke+1)
        candidate_ink=cv2.bitwise_and(dash_mask,(gray[ry0:ry1,rx0:rx1]<180).astype(np.uint8)*255)
        # Thin and faint cadastral strokes may occupy only a few normal samples;
        # a lower evidence threshold protects them conservatively at crossings.
        # Exclude the expected guide stroke, not a broad halo: a thin parcel edge
        # crossing a guide may be only a few pixels wide and still needs protection.
        center_exclusion=np.abs(offsets)<=stroke
        boundary_evidence=values.copy(); boundary_evidence[:,center_exclusion]=False
        boundary_rows=(values.sum(axis=1)>=max(4,boundary_reach//8))&boundary_evidence.any(axis=1)
        if np.any(boundary_rows):
            row_mask=boundary_rows[:,None]&values&(~center_exclusion[None,:])&inside
            crossing=np.zeros_like(dash_mask)
            crossing[ys[row_mask]-ry0,xs[row_mask]-rx0]=255
            crossing=cv2.dilate(crossing,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
            crossing[~local_boundary]=0
            protected[ry0:ry1,rx0:rx1]=cv2.bitwise_or(protected[ry0:ry1,rx0:rx1],crossing)
        local_protected=protected[ry0:ry1,rx0:rx1]
        candidate_ink[local_protected>0]=0
        removal[ry0:ry1,rx0:rx1]=cv2.bitwise_or(removal[ry0:ry1,rx0:rx1],candidate_ink)
        candidate['mask_pixels']=int(cv2.countNonZero(candidate_ink)); candidate['protected_boundary_pixels']=int(cv2.countNonZero(cv2.bitwise_and(dash_mask,local_protected)))
        candidate['review_required']=bool(candidate['protected_boundary_pixels'])
        candidate['review_reason']='guide crosses or approaches protected boundary pixels' if candidate['review_required'] else None
        candidate['classification']='ambiguous-review' if candidate['review_required'] or not candidate['mask_pixels'] else 'high-confidence-removable'
        classified.append(candidate)
    return removal,protected,classified


def build_blue_symbol_mask(rgb: np.ndarray, candidates: list[dict], protected: np.ndarray|None=None) -> tuple[np.ndarray,list[dict]]:
    """Mask only blue ink inside confidently detected hatched boxes, never black geometry."""
    h,w=rgb.shape[:2]; mask=np.zeros((h,w),np.uint8); classified=[]
    for item in candidates:
        candidate=dict(item); x=int(item['x']); y=int(item['y']); bw=int(item['width']); bh=int(item['height'])
        x0=max(0,x-1); y0=max(0,y-1); x1=min(w,x+bw+1); y1=min(h,y+bh+1)
        if item.get('classification')!='high-confidence-removable':
            candidate['classification']='ambiguous-review'; classified.append(candidate); continue
        hsv=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2HSV)
        blue=cv2.inRange(hsv,np.array([82,38,32],np.uint8),np.array([153,255,255],np.uint8))
        if protected is not None: blue[protected[y0:y1,x0:x1]>0]=0
        pixels=int(cv2.countNonZero(blue))
        if pixels<28:
            candidate['classification']='ambiguous-review'; candidate['mask_pixels']=0
        else:
            mask[y0:y1,x0:x1]=cv2.bitwise_or(mask[y0:y1,x0:x1],blue)
            candidate['mask_pixels']=pixels
        classified.append(candidate)
    return mask,classified

import unittest
from pathlib import Path

import cv2
import numpy as np
import uuid
from fastapi import BackgroundTasks, HTTPException

from app.main import apply_removal_mask, build_principal_label_mask, build_red_ink_mask, is_principal_integer, exclude_protected_pixels, _has_decimal_component_signature, map_ocr_box, detect_merged_measurement_labels, detect_color_measurements, detect_oriented_measurements, _ocr_small_region, _write_detection_cache, _cache_path, assets, jobs, PROCESSED_DIR, ORIGINALS_DIR, THUMB_DIR, MANUAL_EDITS_DIR, JOBS_DIR, persist_job, normalize_asset_record, public_asset, list_maps, delete_map, thumbnail, map_preview, job_preview, export_png, approve, edit_candidate, save_manual_edits, get_manual_edits, Approval, process_map, _removal_has_evidence
from app.manual_edits import display_to_source_point


class CleaningPixelTests(unittest.TestCase):
    def test_red_integer_annotations_are_not_protected_as_principal_labels(self):
        image=np.full((100,180,3),255,np.uint8)
        cv2.putText(image,'15',(25,60),cv2.FONT_HERSHEY_SIMPLEX,1,(220,40,40),2,cv2.LINE_AA)
        cv2.putText(image,'12',(105,60),cv2.FONT_HERSHEY_SIMPLEX,1,(30,70,220),2,cv2.LINE_AA)
        red={'text':'15','x':22,'y':35,'width':42,'height':30}
        blue={'text':'12','x':102,'y':35,'width':42,'height':30}
        self.assertFalse(is_principal_integer(image,red))
        self.assertTrue(is_principal_integer(image,blue))
        principal=[b for b in (red,blue) if is_principal_integer(image,b)]
        protected=build_principal_label_mask(image.shape[:2],principal,[],padding=0)
        red_mask=build_red_ink_mask(image)
        self.assertEqual(cv2.countNonZero(cv2.bitwise_and(red_mask,protected)),0)
        self.assertGreater(cv2.countNonZero(protected),0)

    def test_red_mask_selects_annotation_ink_without_nearby_black_boundary(self):
        image=np.full((100,180,3),255,np.uint8)
        cv2.line(image,(10,68),(170,68),(0,0,0),3)
        cv2.putText(image,'64',(60,58),cv2.FONT_HERSHEY_SIMPLEX,.75,(220,40,40),2,cv2.LINE_AA)
        mask=build_red_ink_mask(image)
        self.assertGreater(cv2.countNonZero(mask),20)
        self.assertEqual(cv2.countNonZero(mask[66:71,:]),0,'black boundary entered red mask')

    def test_standalone_integer_labels_survive_complete_mask_combination(self):
        image=np.full((180,360,3),255,np.uint8)
        labels=[('2',(55,62)),('3',(155,62)),('10',(265,62))]
        detections=[]
        for text,origin in labels:
            cv2.putText(image,text,origin,cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,0),2,cv2.LINE_AA)
            (x,y),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.1,2)
            detections.append({'text':text,'x':origin[0]-2,'y':origin[1]-y-2,'width':x+4,'height':y+4,'kind':'possible-principal-label'})
        # Add a genuine parcel edge and a decimal measurement to be removed.
        cv2.line(image,(20,105),(340,105),(0,0,0),3)
        cv2.putText(image,'8.8',(145,145),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
        labels_mask=build_principal_label_mask(image.shape[:2],detections,[])
        boundary_mask=np.zeros(image.shape[:2],np.uint8); cv2.line(boundary_mask,(20,105),(340,105),255,5)
        protected=cv2.bitwise_or(labels_mask,boundary_mask)
        measurement=np.zeros(image.shape[:2],np.uint8); cv2.rectangle(measurement,(132,116),(202,153),255,-1)
        red=np.zeros_like(measurement); cv2.line(red,(45,42),(90,72),255,3)
        guides=np.zeros_like(measurement); cv2.line(guides,(20,105),(340,105),255,3)
        symbols=np.zeros_like(measurement); cv2.rectangle(symbols,(250,38),(300,68),255,-1)
        masks=[exclude_protected_pixels(mask,protected) for mask in (measurement,red,guides,symbols)]
        annotation=cv2.bitwise_or(masks[0],masks[1]); structure=cv2.bitwise_or(masks[2],masks[3])
        candidate=apply_removal_mask(image,annotation,dilation=2,protected=protected)
        candidate=apply_removal_mask(candidate,structure,dilation=0,protected=protected)
        for text,(x,y) in labels:
            (w,h),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,1.1,2)
            self.assertTrue(np.array_equal(image[y-h-4:y+3,x-3:x+w+4],candidate[y-h-4:y+3,x-3:x+w+4]),f'principal label {text} changed')
        self.assertTrue(np.array_equal(image[102:109,20:340],candidate[102:109,20:340]),'parcel boundary changed')
        self.assertGreater(np.count_nonzero(np.any(image[116:153,132:202]!=candidate[116:153,132:202],axis=2)),20,'decimal measurement was not cleaned')

    def test_integer_groups_are_filtered_before_color_ocr_but_decimal_geometry_is_retained(self):
        integer=[(0,0,10,20,45),(12,0,10,20,43)]
        decimal=[(0,0,10,20,45),(12,14,3,3,4),(18,0,10,20,43)]
        self.assertFalse(_has_decimal_component_signature(integer))
        self.assertTrue(_has_decimal_component_signature(decimal))

    def test_candidate_without_removal_evidence_is_not_verified(self):
        self.assertFalse(_removal_has_evidence({'classification':'ambiguous-review','changed_pixels':12}))
        self.assertFalse(_removal_has_evidence({'classification':'high-confidence-removable','mask_pixels':12}))
        self.assertFalse(_removal_has_evidence({'mask_pixels':0,'changed_pixels':12}))
        self.assertTrue(_removal_has_evidence({'mask_pixels':12,'changed_pixels':3}))

    def test_job_state_is_persisted_without_runtime_only_fields(self):
        job_id='persist-'+uuid.uuid4().hex
        jobs[job_id]={'id':job_id,'status':'processing','stage':'inpainting','_stage_started':{'at':1.0},'_started_perf':1.0}
        path=JOBS_DIR/f'{job_id}.json'
        try:
            persist_job(job_id)
            saved=__import__('json').loads(path.read_text())
            self.assertEqual(saved['status'],'processing')
            self.assertNotIn('_stage_started',saved)
            self.assertNotIn('_started_perf',saved)
        finally:
            jobs.pop(job_id,None); path.unlink(missing_ok=True)

    def test_maps_endpoint_handles_legacy_asset_without_filename(self):
        asset_id='legacy-'+uuid.uuid4().hex
        image_path=PROCESSED_DIR/f'{asset_id}.png'
        self.assertTrue(cv2.imwrite(str(image_path),np.full((32,48,3),255,np.uint8)))
        assets[asset_id]={'id':asset_id,'source_path':str(image_path),'approved':False}
        try:
            result=next(item for item in list_maps() if item['id']==asset_id)
            self.assertIsNone(result['filename'])
            self.assertIsNone(result['map_type'])
            self.assertEqual((result['width'],result['height']),(48,32))
            self.assertIn('Original upload filename is unavailable',result['metadata_warnings'][1])
            self.assertEqual(result['image_url'],f'/api/maps/{asset_id}/image')
        finally:
            assets.pop(asset_id,None); image_path.unlink(missing_ok=True)

    def test_legacy_record_recovers_source_by_existing_id_without_fabricating_filename(self):
        asset_id='recover-'+uuid.uuid4().hex
        image_path=ORIGINALS_DIR/f'{asset_id}.tif'
        from PIL import Image
        Image.fromarray(np.full((25,41,3),255,np.uint8)).save(image_path)
        try:
            result=normalize_asset_record({'id':asset_id,'filename':'','map_type':'invalid'})
            self.assertEqual(result['id'],asset_id)
            self.assertEqual(Path(result['source_path']),image_path.resolve())
            self.assertEqual((result['width'],result['height']),(41,25))
            self.assertIsNone(result['filename'])
            self.assertIsNone(result['map_type'])
            malformed=public_asset({'approved':False})
            self.assertIsNone(malformed['id'])
            self.assertIsNone(malformed['image_url'])
            self.assertTrue(malformed['metadata_warnings'])
        finally:
            image_path.unlink(missing_ok=True)

    def test_delete_removes_only_selected_asset_and_its_generated_files(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from PIL import Image
        with TemporaryDirectory() as root:
            root=Path(root); dirs={name:root/name for name in ('originals','thumbnails','processed','jobs','manual_edits','projects')}
            for directory in dirs.values(): directory.mkdir()
            asset_id='delete-'+uuid.uuid4().hex; job_id='job-'+uuid.uuid4().hex; other_id='keep-'+uuid.uuid4().hex
            source=dirs['originals']/f'{asset_id}.tif'; Image.fromarray(np.full((30,45,3),255,np.uint8)).save(source)
            thumb=dirs['thumbnails']/f'{asset_id}.png'; thumb.write_bytes(b'thumb')
            owned=[dirs['processed']/f'{job_id}.png',dirs['processed']/f'{job_id}-mask.png',dirs['jobs']/f'{job_id}.json',dirs['manual_edits']/f'{job_id}.json']
            for path in owned: path.write_bytes(b'artifact')
            other_source=dirs['originals']/f'{other_id}.png'; Image.fromarray(np.full((20,25,3),255,np.uint8)).save(other_source)
            unrelated=dirs['processed']/f'{other_id}-candidate.png'; unrelated.write_bytes(b'keep')
            original_dirs=(ORIGINALS_DIR,)
            assets[asset_id]={'id':asset_id,'filename':'known-sheet.tif','map_type':'sheet','width':45,'height':30,'source_path':str(source)}
            assets[other_id]={'id':other_id,'filename':'other.png','map_type':'sheet','width':25,'height':20,'source_path':str(other_source)}
            jobs[job_id]={'id':job_id,'asset_id':asset_id,'status':'completed'}
            try:
                patches=[patch('app.main.ORIGINALS_DIR',dirs['originals']),patch('app.main.THUMB_DIR',dirs['thumbnails']),patch('app.main.PROCESSED_DIR',dirs['processed']),patch('app.main.JOBS_DIR',dirs['jobs']),patch('app.main.MANUAL_EDITS_DIR',dirs['manual_edits']),patch('app.main.PROJECTS_DIR',dirs['projects']),patch('app.main.persist_assets')]
                for item in patches: item.start()
                result=delete_map(asset_id)
                self.assertTrue(result['deleted']); self.assertEqual(result['jobs_removed'],1)
                self.assertFalse(source.exists()); self.assertFalse(thumb.exists())
                self.assertTrue(all(not path.exists() for path in owned))
                self.assertTrue(other_source.exists()); self.assertTrue(unrelated.exists())
                self.assertIn(other_id,assets); self.assertNotIn(asset_id,assets); self.assertNotIn(job_id,jobs)
            finally:
                for item in locals().get('patches',[]): item.stop()
                assets.pop(asset_id,None); assets.pop(other_id,None); jobs.pop(job_id,None)

    def test_delete_missing_id_returns_404(self):
        with self.assertRaises(HTTPException) as ctx: delete_map('missing-'+uuid.uuid4().hex)
        self.assertEqual(ctx.exception.status_code,404)

    def test_delete_is_blocked_when_assembly_references_asset(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from PIL import Image
        with TemporaryDirectory() as root:
            root=Path(root); originals=root/'originals'; projects_dir=root/'projects'; originals.mkdir(); projects_dir.mkdir()
            asset_id='referenced-'+uuid.uuid4().hex; source=originals/f'{asset_id}.png'; Image.fromarray(np.full((20,20,3),255,np.uint8)).save(source)
            (projects_dir/'assembly.json').write_text(__import__('json').dumps({'layout':{'primary':asset_id}}))
            assets[asset_id]={'id':asset_id,'filename':'map.png','map_type':'primary','width':20,'height':20,'source_path':str(source)}
            try:
                with patch('app.main.ORIGINALS_DIR',originals),patch('app.main.PROJECTS_DIR',projects_dir):
                    with self.assertRaises(HTTPException) as ctx: delete_map(asset_id)
                self.assertEqual(ctx.exception.status_code,409); self.assertTrue(source.exists()); self.assertIn(asset_id,assets)
            finally: assets.pop(asset_id,None)

    def test_corrupt_thumbnail_is_regenerated_and_map_preview_falls_back_to_original(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from PIL import Image
        with TemporaryDirectory() as directory:
            root=Path(directory); originals=root/'originals'; thumbs=root/'thumbs'; processed=root/'processed'
            originals.mkdir(); thumbs.mkdir(); processed.mkdir(); asset_id='thumb-'+uuid.uuid4().hex
            source=originals/f'{asset_id}.tif'; Image.fromarray(np.full((48,72,3),220,np.uint8)).save(source)
            thumb_path=thumbs/f'{asset_id}.png'; thumb_path.write_bytes(b'corrupt thumbnail')
            assets[asset_id]={'id':asset_id,'filename':'known.tif','map_type':'sheet','width':72,'height':48,'source_path':str(source)}
            try:
                with patch('app.main.THUMB_DIR',thumbs),patch('app.main.ORIGINALS_DIR',originals),patch('app.main.PROCESSED_DIR',processed):
                    thumb_response=thumbnail(asset_id); preview_response=map_preview(asset_id)
                self.assertEqual(thumb_response.media_type,'image/png')
                self.assertEqual(preview_response.headers['X-Preview-Fallback'],'original')
                with Image.open(thumb_path) as recovered: recovered.verify()
            finally: assets.pop(asset_id,None); thumb_path.unlink(missing_ok=True); source.unlink(missing_ok=True)

    def test_display_to_source_mapping_at_zoom_and_rotation(self):
        for angle in (0, 35, 90, -125):
            dw,dh,left,top=400.0,200.0,30.0,50.0
            theta=np.deg2rad(angle); bw=abs(dw*np.cos(theta))+abs(dh*np.sin(theta)); bh=abs(dw*np.sin(theta))+abs(dh*np.cos(theta))
            px,py=200.0,100.0; ux=px*dw/800; uy=py*dh/400
            dx=ux-dw/2; dy=uy-dh/2
            sx=dx*np.cos(theta)-dy*np.sin(theta); sy=dx*np.sin(theta)+dy*np.cos(theta)
            client_x=left+bw/2+sx; client_y=top+bh/2+sy
            self.assertEqual(display_to_source_point(client_x,client_y,left,top,dw,dh,800,400,angle),(200,100))

    def test_manual_edits_persist_version_invalidate_approval_and_keep_source_immutable(self):
        asset_id='manual-'+uuid.uuid4().hex; job_id='manual-'+uuid.uuid4().hex
        image=np.full((48,72,3),255,np.uint8); image[20:23,5:65]=0
        source=PROCESSED_DIR/f'{asset_id}-source.png'; base=PROCESSED_DIR/f'{job_id}.png'
        self.assertTrue(cv2.imwrite(str(source),image)); self.assertTrue(cv2.imwrite(str(base),image))
        import hashlib
        original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        assets[asset_id]={'id':asset_id,'width':72,'height':48,'approved':False}
        jobs[job_id]={'id':job_id,'asset_id':asset_id,'status':'completed','preview_available':True,'candidate_version':0,'candidate_path':str(base),'manual_operations':[]}
        try:
            from unittest.mock import patch
            with patch('app.main.persist_assets'), patch('app.main.persist_job'):
                result=save_manual_edits(job_id,{'candidate_version':0,'operations':[{'tool':'erase','points':[[20,21]],'width':9,'color':'#ffffff'}]})
                self.assertEqual(result['candidate_version'],1)
                self.assertEqual(get_manual_edits(job_id)['operations'][0]['tool'],'erase')
                first=Path(jobs[job_id]['candidate_path']); edited=cv2.imread(str(first)); self.assertFalse(np.array_equal(image[15:28,15:27],edited[15:28,15:27]))
                approve(job_id,Approval(approved=True))
                save_manual_edits(job_id,{'candidate_version':1,'operations':[]})
                self.assertFalse(jobs[job_id]['approved'])
                with self.assertRaises(HTTPException): export_png(job_id)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),original_hash)
            self.assertTrue(Path(MANUAL_EDITS_DIR/f'{job_id}.json').exists())
        finally:
            assets.pop(asset_id,None); jobs.pop(job_id,None)
            for p in PROCESSED_DIR.glob(f'{job_id}*'): p.unlink(missing_ok=True)
            (MANUAL_EDITS_DIR/f'{job_id}.json').unlink(missing_ok=True); source.unlink(missing_ok=True)

    def test_detector_cache_reuses_results_for_same_source_configuration(self):
        key='unit-cache-' + uuid.uuid4().hex
        expected=[{'text':'39.2','x':10,'y':20,'width':50,'height':24,'confidence':95}]
        cache_file=_cache_path(key,'color-measurements')
        try:
            _write_detection_cache(key,'color-measurements',expected)
            result=detect_color_measurements(np.full((64,96,3),255,np.uint8),cache_key=key)
            self.assertEqual(result,expected)
        finally:
            cache_file.unlink(missing_ok=True)

    def test_active_processing_request_reuses_existing_background_job(self):
        asset_id = 'test-' + uuid.uuid4().hex
        assets[asset_id] = {'id': asset_id, 'source_path': '/nonexistent/source.tif'}
        tasks = BackgroundTasks()
        try:
            from unittest.mock import patch
            with patch('app.main.persist_job'), patch('app.main.persist_assets'):
                first = process_map(asset_id, tasks)
                second = process_map(asset_id, BackgroundTasks())
            self.assertEqual(first['job_id'], second['job_id'])
            self.assertTrue(second['reused'])
            self.assertEqual(len(tasks.tasks), 1)
        finally:
            assets.pop(asset_id, None)
            jobs.pop(first['job_id'], None)

    def test_measurement_pixels_change_plot_label_and_boundary_remain(self):
        image = np.full((260, 420, 3), 255, np.uint8)
        cv2.rectangle(image, (15, 15), (404, 244), (0, 0, 0), 2)
        cv2.line(image, (15, 130), (404, 130), (0, 0, 0), 2)
        cv2.putText(image, '7', (90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, '62.4', (210, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
        source = image.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(mask, (205, 88), (305, 119), 255, -1)
        cleaned = apply_removal_mask(image, mask, dilation=2)
        self.assertGreater(int(np.count_nonzero(np.any(source[88:120, 205:306] != cleaned[88:120, 205:306], axis=2))), 100)
        self.assertTrue(np.array_equal(source[55:105, 75:115], cleaned[55:105, 75:115]), 'principal label was modified')
        self.assertTrue(np.array_equal(source[20:240, 20:25], cleaned[20:240, 20:25]), 'outer boundary was modified')
        self.assertTrue(np.array_equal(source[128:133, 30:190], cleaned[128:133, 30:190]), 'internal boundary was modified')

    def test_ocr_box_inverse_rotation_and_scale_stay_in_image(self):
        # Source bbox x=25,y=40,w=30,h=12 on a 200x100 image, viewed at 2x then CCW-rotated.
        expected = (25, 40, 30, 12)
        rotated_box = (2 * 40, 2 * (200 - 25 - 30), 2 * 12, 2 * 30)
        self.assertEqual(map_ocr_box(*rotated_box, 90, 200, 100, 2), expected)
        self.assertEqual(map_ocr_box(*rotated_box, 90, 200, 100, 2)[0] < 200, True)

    def test_adjacent_37_6_and_22_are_split_and_only_measurement_is_removed(self):
        image = np.full((280, 520, 3), 255, np.uint8)
        blue = (0, 0, 230)
        cv2.putText(image, '37.6', (35, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, blue, 3, cv2.LINE_AA)
        cv2.putText(image, '22', (140, 155), cv2.FONT_HERSHEY_DUPLEX, 2.0, blue, 3, cv2.LINE_AA)
        cv2.line(image, (10, 192), (510, 192), (0, 0, 0), 10)
        for x in range(360, 500, 32):
            cv2.line(image, (x, 220), (x + 13, 233), (0, 0, 0), 2)

        matches = detect_merged_measurement_labels(image, tile_size=520)
        self.assertTrue(any(m['text'] == '37.6' and m['protected_text'] == '22' for m in matches), matches)
        item = next(m for m in matches if m['text'] == '37.6' and m['protected_text'] == '22')
        measurement = np.zeros(image.shape[:2], np.uint8)
        x,y,w,h=[item[k] for k in ('x','y','width','height')]
        cv2.rectangle(measurement,(x-1,y-1),(x+w+1,y+h+1),255,-1)
        protected = np.zeros(image.shape[:2], np.uint8)
        pb=item['protected_box']; cv2.rectangle(protected,(pb['x']-2,pb['y']-2),(pb['x']+pb['width']+2,pb['y']+pb['height']+2),255,-1)
        cleaned = apply_removal_mask(image, measurement, dilation=2, protected=protected)
        self.assertGreater(np.count_nonzero(np.any(image[65:130,30:140] != cleaned[65:130,30:140],axis=2)), 100)
        px,py,pw,ph=[item['protected_box'][k] for k in ('x','y','width','height')]
        self.assertTrue(np.array_equal(image[py:py+ph,px:px+pw], cleaned[py:py+ph,px:px+pw]), 'principal number 22 changed')
        self.assertEqual(_ocr_small_region(cleaned,px-5,py-5,px+pw+5,py+ph+5)[0], '22')
        self.assertTrue(np.array_equal(image[187:198,:], cleaned[187:198,:]), 'thick parcel boundary changed')

    def test_blue_39_2_detected_and_removed_near_protected_15_and_boundary(self):
        image = np.full((260, 620, 3), 255, np.uint8)
        blue = (0, 0, 230)
        cv2.putText(image, '39.2', (195, 118), cv2.FONT_HERSHEY_DUPLEX, 1.35, blue, 2, cv2.LINE_AA)
        cv2.putText(image, '15', (355, 150), cv2.FONT_HERSHEY_DUPLEX, 1.5, blue, 2, cv2.LINE_AA)
        cv2.line(image, (20, 145), (600, 145), (0, 0, 0), 4)
        source=image.copy()
        matches,integer_labels=detect_color_measurements(image,tile_size=620,include_labels=True)
        item=next((m for m in matches if m['text']=='39.2'),None)
        self.assertIsNotNone(item,matches)
        self.assertTrue(any(b['text']=='15' for b in integer_labels),integer_labels)
        mask=np.zeros(image.shape[:2],np.uint8)
        x,y,w,h=[item[k] for k in ('x','y','width','height')]
        cv2.rectangle(mask,(max(0,x-3),max(0,y-3)),(min(image.shape[1]-1,x+w+3),min(image.shape[0]-1,y+h+3)),255,-1)
        # Leave clearance around the thick genuine boundary and principal plot label.
        protected=build_principal_label_mask(image.shape[:2],integer_labels,[],padding=2)
        cleaned=apply_removal_mask(image,mask,dilation=1,protected=protected)
        self.assertGreater(np.count_nonzero(np.any(source[y:y+h,x:x+w] != cleaned[y:y+h,x:x+w],axis=2)),30)
        self.assertEqual(_ocr_small_region(source,330,100,410,170)[0],'15')
        self.assertEqual(_ocr_small_region(cleaned,330,100,410,170)[0],'15')
        self.assertTrue(np.array_equal(source[142:149,20:600],cleaned[142:149,20:600]),'nearby parcel boundary changed')

    def test_rotated_blue_and_horizontal_black_decimal_detection(self):
        for ink,rotation in (((0,0,230),cv2.ROTATE_90_CLOCKWISE),((0,0,0),None)):
            image=np.full((220,400,3),255,np.uint8)
            cv2.putText(image,'45.6',(100,105),cv2.FONT_HERSHEY_DUPLEX,1.3,ink,2,cv2.LINE_AA)
            if rotation is not None: image=cv2.rotate(image,rotation)
            matches=detect_color_measurements(image,tile_size=500)
            found=next((b for b in matches if b['text']=='45.6'),None)
            self.assertIsNotNone(found,matches)
            self.assertGreater(found['width'],20); self.assertGreater(found['height'],20)
            self.assertLessEqual(found['x']+found['width'],image.shape[1])
            self.assertLessEqual(found['y']+found['height'],image.shape[0])

    def test_small_black_and_vertical_blue_decimals_are_removed_while_label_and_boundary_survive(self):
        for value,color,angle,origin in (('3.0',(0,0,0),0,(150,120)),('9.0',(0,0,230),90,(250,150)),('10.0',(0,0,0),35,(250,150))):
            with self.subTest(value=value):
                image=np.full((320,520,3),255,np.uint8)
                cv2.putText(image,value,origin,cv2.FONT_HERSHEY_SIMPLEX,.62,color,1,cv2.LINE_AA)
                if angle:
                    transform=cv2.getRotationMatrix2D((260,150),angle,1)
                    image=cv2.warpAffine(image,transform,(520,320),borderValue=(255,255,255))
                cv2.putText(image,'15',(365,145),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,0,230),2,cv2.LINE_AA)
                cv2.line(image,(20,270),(500,270),(0,0,0),4)
                source=image.copy()
                found=detect_color_measurements(image,tile_size=620)
                found+=detect_oriented_measurements(image,tile_size=620)
                item=next((b for b in found if '.' in b['text'] or ',' in b['text']),None)
                self.assertIsNotNone(item,found)
                mask=np.zeros(image.shape[:2],np.uint8)
                components=item.get('component_boxes') or [item]
                for component in components:
                    x,y,w,h=[int(component[k]) for k in ('x','y','width','height')]
                    cv2.rectangle(mask,(max(0,x-2),max(0,y-2)),(min(519,x+w+2),min(319,y+h+2)),255,-1)
                cleaned=apply_removal_mask(image,mask,dilation=1)
                x,y,w,h=[int(item[k]) for k in ('x','y','width','height')]
                self.assertGreater(np.count_nonzero(np.any(source[y:y+h,x:x+w]!=cleaned[y:y+h,x:x+w],axis=2)),3)
                self.assertEqual(_ocr_small_region(cleaned,350,90,420,160)[0],'15')
                self.assertTrue(np.array_equal(source[268:273,:],cleaned[268:273,:]))

    def test_requested_decimal_families_are_masked_without_harming_plot_id_or_boundary(self):
        cases=(('8.8',0),('13.4',-65),('59.2',-65),('62.6',-45),('(225.4)',-65))
        for value,angle in cases:
            with self.subTest(value=value):
                image=np.full((600,760,3),255,np.uint8); blue=(0,0,230)
                layer=np.full((360,560,3),255,np.uint8)
                cv2.putText(layer,value,(80,230),cv2.FONT_HERSHEY_DUPLEX,1.45,blue,2,cv2.LINE_AA)
                if angle:
                    matrix=cv2.getRotationMatrix2D((layer.shape[1]/2,layer.shape[0]/2),angle,1)
                    layer=cv2.warpAffine(layer,matrix,(layer.shape[1],layer.shape[0]),borderValue=(255,255,255))
                image[100:460,100:660]=layer
                cv2.putText(image,'15',(585,145),cv2.FONT_HERSHEY_DUPLEX,1.15,blue,2,cv2.LINE_AA)
                cv2.line(image,(20,550),(740,550),(0,0,0),4)
                source=image.copy()
                matches=detect_color_measurements(image,tile_size=900)
                matches+=detect_oriented_measurements(image,tile_size=900)
                item=next((b for b in matches if b['text'].strip('()[]')==value.strip('()[]')),None)
                self.assertIsNotNone(item,matches)
                mask=np.zeros(image.shape[:2],np.uint8); pad=3
                for c in item.get('component_boxes',[]):
                    cv2.rectangle(mask,(max(0,c['x']-pad),max(0,c['y']-pad)),(min(759,c['x']+c['width']+pad),min(519,c['y']+c['height']+pad)),255,-1)
                if not item.get('component_boxes'):
                    x,y,w,h=[item[k] for k in ('x','y','width','height')]
                    cv2.rectangle(mask,(x-pad,y-pad),(x+w+pad,y+h+pad),255,-1)
                cleaned=apply_removal_mask(image,mask,dilation=2)
                x,y,w,h=[item[k] for k in ('x','y','width','height')]
                self.assertGreater(np.count_nonzero(np.any(source[y:y+h,x:x+w]!=cleaned[y:y+h,x:x+w],axis=2)),30)
                self.assertEqual(_ocr_small_region(cleaned,565,95,655,155)[0],'15')
                self.assertTrue(np.array_equal(source[548:553,:],cleaned[548:553,:]),'parcel boundary changed')

    def test_preview_serves_latest_candidate_and_export_requires_approval(self):
        asset_id = 'test-' + uuid.uuid4().hex
        job_id = 'test-' + uuid.uuid4().hex
        candidate = np.full((24, 32, 3), 255, np.uint8)
        candidate[5:12, 8:19] = (12, 40, 180)
        artifact = PROCESSED_DIR / f'{job_id}.png'
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(candidate, cv2.COLOR_RGB2BGR))
        self.assertTrue(ok)
        artifact.write_bytes(encoded.tobytes())
        assets[asset_id] = {'id': asset_id, 'latest_job_id': job_id, 'approved': False}
        jobs[job_id] = {'id': job_id, 'asset_id': asset_id, 'status': 'completed', 'preview_available': True, 'approved': False}
        try:
            from unittest.mock import patch
            # These endpoints persist the asset catalog in production. Keep this
            # synthetic record isolated so the test cannot overwrite user data.
            with patch('app.main.persist_assets'), patch('app.main.persist_job'):
                self.assertEqual(job_preview(job_id).body, artifact.read_bytes())
                self.assertEqual(map_preview(asset_id).body, artifact.read_bytes())
                with self.assertRaises(HTTPException) as blocked:
                    export_png(job_id)
                self.assertEqual(blocked.exception.status_code, 403)
                approve(job_id, Approval(approved=True))
                self.assertEqual(export_png(job_id).body, artifact.read_bytes())
                edit_candidate(job_id, {'edited': True})
                with self.assertRaises(HTTPException) as invalidated:
                    export_png(job_id)
                self.assertEqual(invalidated.exception.status_code, 403)
        finally:
            assets.pop(asset_id, None); jobs.pop(job_id, None); artifact.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()

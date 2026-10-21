import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import cv2
import numpy as np
from PIL import Image
from app.main import apply_removal_mask, detect_color_measurements, detect_oriented_measurements
from app.shape_analysis import detect_box_symbols, detect_dashed_guides, build_guide_removal_mask, build_blue_symbol_mask

class ShapeAnalysisTests(unittest.TestCase):
    def test_short_diagonal_dash_train_is_detected_and_removed_without_touching_plot_or_boundary(self):
        image=np.full((260,520,3),255,np.uint8)
        for x in range(24,470,27):
            y=28+round((x-24)*.24)
            cv2.line(image,(x,y),(x+11,y+3),(0,0,0),2)
        cv2.line(image,(14,204),(506,204),(0,0,0),3)
        cv2.putText(image,'15',(280,152),cv2.FONT_HERSHEY_SIMPLEX,1.15,(0,0,0),2,cv2.LINE_AA)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        labels=np.zeros(gray.shape,np.uint8); labels[114:159,270:320]=255
        candidates=detect_dashed_guides(gray,tile_size=520,rgb=image)
        train=next((q for q in candidates if q.get('orientation')=='diagonal'
                    and q.get('classification')=='high-confidence-removable'),None)
        self.assertIsNotNone(train,candidates)
        mask,protected,classified=build_guide_removal_mask(gray,[train],image,labels)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected)
        self.assertGreater(cv2.countNonZero(mask),40)
        self.assertTrue(np.array_equal(image[114:159,270:320],cleaned[114:159,270:320]),'plot number changed')
        self.assertTrue(np.array_equal(image[200:209,:],cleaned[200:209,:]),'parcel boundary changed')
        self.assertGreater(np.count_nonzero(np.any(image[:150]!=cleaned[:150],axis=2)),50)
        self.assertGreater(classified[0]['mask_pixels'],0)

    def test_component_dash_train_removes_only_diagonal_ink_and_protects_label_and_boundary(self):
        image=np.full((280,520,3),255,np.uint8)
        # Diagonal dash train; the solid cadastral edge and principal integer are
        # nearby but are not part of the train.
        for x in range(24,465,60):
            y=35+round((x-24)*.22)
            cv2.line(image,(x,y),(x+40,y+9),(0,0,0),2)
        cv2.line(image,(12,220),(508,220),(0,0,0),3)
        cv2.putText(image,'15',(265,157),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,0),2,cv2.LINE_AA)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        labels=np.zeros(gray.shape,np.uint8); labels[118:164,250:300]=255
        candidates=detect_dashed_guides(gray,tile_size=520,rgb=image)
        train=next((q for q in candidates if q.get('detection_method')=='collinear-connected-components'),None)
        self.assertIsNotNone(train,candidates)
        mask,_,classified=build_guide_removal_mask(gray,[train],image,labels)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=labels)
        self.assertGreater(cv2.countNonZero(mask),100)
        self.assertTrue(np.array_equal(image[216:225,:],cleaned[216:225,:]),'parcel boundary changed')
        self.assertTrue(np.array_equal(image[118:164,250:300],cleaned[118:164,250:300]),'principal plot number changed')
        self.assertGreater(np.count_nonzero(np.any(image[25:145,:,:]!=cleaned[25:145,:,:],axis=2)),100)
        self.assertTrue(classified[0]['mask_pixels']>0)


    def test_compact_diagonal_dash_train_masks_only_observed_marks(self):
        image = np.full((300, 440, 3), 255, np.uint8)
        dash_boxes = [(40, 30), (54, 60), (68, 90), (82, 120), (96, 150)]
        for index, (x, y) in enumerate(dash_boxes):
            if index == len(dash_boxes) - 1:
                # Include a compact mark that is only 7 pixels tall.
                cv2.rectangle(image, (x, y + 1), (x + 9, y + 7), (0, 0, 0), -1)
            else:
                cv2.rectangle(image, (x, y), (x + 9, y + 9), (0, 0, 0), -1)

        cv2.line(image, (12, 265), (428, 265), (0, 0, 0), 3)
        cv2.putText(
            image, "15", (230, 200), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 0), 2, cv2.LINE_AA,
        )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        labels = np.zeros(gray.shape, np.uint8)
        labels[170:212, 220:320] = 255

        candidates = detect_dashed_guides(gray, tile_size=440, rgb=image)
        train = next((item for item in candidates
            if item.get("detection_method") == "collinear-connected-components"
            and item.get("orientation") == "diagonal"
            and item.get("dash_runs", 0) >= 4
            and 30 <= item["x1"] <= 60
            and 20 <= item["y1"] <= 50
        ), None)
        self.assertIsNotNone(train, candidates)
        self.assertGreaterEqual(len(train.get("component_boxes", [])), 4)

        mask, protected, classified = build_guide_removal_mask(gray, [train], image, labels)
        combined_protection = cv2.bitwise_or(protected, labels)
        cleaned = apply_removal_mask(image, mask, dilation=0, protected=combined_protection)

        self.assertGreater(cv2.countNonZero(mask), 100)
        self.assertTrue(np.array_equal(image[261:270, :], cleaned[261:270, :]), "solid parcel boundary changed")
        self.assertTrue(np.array_equal(image[170:212, 220:320], cleaned[170:212, 220:320]), "principal label changed")
        for x, y in dash_boxes:
            changed = np.count_nonzero(np.any(image[y:y + 10, x:x + 10] != cleaned[y:y + 10, x:x + 10], axis=2))
            self.assertGreater(changed, 40, f"dash at {(x, y)} was not removed")
        self.assertGreater(classified[0]["mask_pixels"], 0)


    def test_aligned_solid_stroke_is_attached_to_compact_diagonal_dash_train(self):
        image = np.full((330, 460, 3), 255, np.uint8)

        # An isolated thin solid stroke followed by five detached compact dashes.
        cv2.line(image, (10, 10), (55, 105), (0, 0, 0), 3)
        dash_boxes = [(60, 115), (74, 145), (88, 175), (102, 205), (116, 235)]
        for x, y in dash_boxes:
            cv2.rectangle(image, (x, y), (x + 9, y + 9), (0, 0, 0), -1)

        # Unrelated boundary and protected label must remain byte-identical.
        cv2.line(image, (12, 305), (448, 305), (0, 0, 0), 3)
        cv2.putText(
            image, "15", (252, 282), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 0, 0), 2, cv2.LINE_AA,
        )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        labels = np.zeros(gray.shape, np.uint8)
        labels[252:289, 245:320] = 255

        candidates = detect_dashed_guides(gray, tile_size=460, rgb=image)
        train = next((item for item in candidates
            if item.get("detection_method") == "collinear-connected-components"
            and item.get("orientation") == "diagonal"
            and item.get("dash_runs", 0) >= 4
            and item.get("component_boxes")
            and 45 <= item["x1"] <= 75
            and 100 <= item["y1"] <= 135
        ), None)
        self.assertIsNotNone(train, candidates)
        self.assertEqual(train.get("attached_solid_component_count"), 1, train)
        self.assertEqual(len(train["component_boxes"]), 6, train["component_boxes"])

        mask, protected, classified = build_guide_removal_mask(
            gray, [train], image, labels
        )
        combined_protection = cv2.bitwise_or(protected, labels)
        cleaned = apply_removal_mask(
            image, mask, dilation=0, protected=combined_protection
        )

        self.assertGreater(cv2.countNonZero(mask), 250)
        self.assertGreater(
            np.count_nonzero(np.any(image[8:108, 8:60] != cleaned[8:108, 8:60], axis=2)),
            100,
            "aligned solid guide stroke was not removed",
        )
        for x, y in dash_boxes:
            self.assertGreater(
                np.count_nonzero(np.any(image[y:y + 10, x:x + 10] != cleaned[y:y + 10, x:x + 10], axis=2)),
                40,
                f"dash at {(x, y)} was not removed",
            )
        self.assertTrue(
            np.array_equal(image[301:310, :], cleaned[301:310, :]),
            "solid parcel boundary changed",
        )
        self.assertTrue(
            np.array_equal(image[252:289, 245:320], cleaned[252:289, 245:320]),
            "protected label changed",
        )
        self.assertGreater(classified[0]["mask_pixels"], 0)

    def test_real_manur_dash_crop_selects_guide_ink_and_preserves_boundary_and_plot_labels(self):
        fixture=Path(__file__).parent/'fixtures'/'manur-dash-top-crop.png'
        image=np.asarray(Image.open(fixture).convert('RGB'))
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        candidates=detect_dashed_guides(gray,rgb=image)
        trains=[q for q in candidates if q.get('detection_method')=='collinear-connected-components']
        self.assertTrue(trains,'real crop dash train was not found')
        mask,protected,classified=build_guide_removal_mask(gray,trains,image)
        self.assertGreater(cv2.countNonZero(mask),1000)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected)
        # The source crop includes a thick cadastral corner/intersection and red
        # integer labels. Guide pixels at the fused intersection remain review-only.
        self.assertTrue(np.array_equal(image[35:65,2055:2105],cleaned[35:65,2055:2105]),'outer boundary changed')
        self.assertTrue(np.array_equal(image[40:400,2055:2105],cleaned[40:400,2055:2105]),'vertical parcel boundary changed')
        self.assertTrue(np.array_equal(image[170:230,1960:2040],cleaned[170:230,1960:2040]),'plot number changed')
        label_roi=image[150:265,620:900]; clean_roi=cleaned[150:265,620:900]
        chroma=label_roi.max(axis=2).astype(np.int16)-label_roi.min(axis=2).astype(np.int16)
        self.assertTrue(np.array_equal(label_roi[chroma>=38],clean_roi[chroma>=38]),'colored text pixels changed')
        self.assertTrue(any(q.get('review_required') for q in classified),'boundary interruption should remain flagged for review')

    def test_real_manur_boundary_adjacent_guide_keeps_boundary_and_colored_plot_labels(self):
        fixture=Path(__file__).parent/'fixtures'/'manur-dash-boundary-crop.png'
        image=np.asarray(Image.open(fixture).convert('RGB'))
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        candidates=detect_dashed_guides(gray,rgb=image)
        trains=[q for q in candidates if q.get('detection_method')=='collinear-connected-components']
        self.assertTrue(trains,'boundary-adjacent dash train was not found')
        mask,protected,classified=build_guide_removal_mask(gray,trains,image)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected)
        # Reference the thick cadastral edge in this test fixture only. The guide
        # runs alongside it; every edge pixel must remain byte-identical.
        boundary=np.zeros(gray.shape,np.uint8)
        cv2.line(boundary,(1200,0),(620,1730),255,15)
        self.assertGreater(cv2.countNonZero(mask),1000)
        self.assertEqual(cv2.countNonZero(cv2.bitwise_and(mask,boundary)),0,'guide mask touched a genuine parcel boundary')
        self.assertTrue(np.array_equal(image[boundary>0],cleaned[boundary>0]),'genuine boundary pixels changed')
        color=image.max(axis=2).astype(np.int16)-image.min(axis=2).astype(np.int16)
        for x0,y0,x1,y1 in ((540,1720,650,1850),(1000,0,1300,180)):
            roi=image[y0:y1,x0:x1]; selected=color[y0:y1,x0:x1]>=38
            self.assertTrue(np.array_equal(roi[selected],cleaned[y0:y1,x0:x1][selected]),'colored parcel label changed')
        self.assertTrue(any(q['mask_pixels']>0 for q in classified))

    def test_box_symbol_is_proposed_but_solid_boundary_is_not_guide(self):
        image=np.full((240,360,3),255,np.uint8)
        cv2.rectangle(image,(25,25),(330,215),(0,0,0),3)
        cv2.rectangle(image,(110,80),(155,125),(0,0,0),2)
        boxes=detect_box_symbols(image,tile_size=256)
        self.assertTrue(any(90<=b['x']<=115 and 60<=b['y']<=90 for b in boxes),boxes)
        guides=detect_dashed_guides(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),tile_size=256)
        self.assertFalse(any(abs(q['y1']-q['y2'])<2 and q['x2']-q['x1']>100 for q in guides),guides)

    def test_periodic_dash_train_is_proposed_for_review(self):
        image=np.full((180,320),255,np.uint8)
        for x in range(20,270,24): cv2.line(image,(x,85),(x+12,85),0,2)
        candidates=detect_dashed_guides(image,tile_size=256)
        self.assertTrue(candidates,candidates)
        self.assertTrue(all(q['kind']=='periodic-line-review' for q in candidates))

    def test_three_well_spaced_dashes_are_removed_but_a_nearly_solid_edge_is_not(self):
        image=np.full((180,340),255,np.uint8)
        for x in range(24,310,23): cv2.line(image,(x,46),(x+10,46),0,2)
        # A legitimate solid boundary has tiny breaks but remains continuously
        # occupied; it must not be reclassified as a construction guide.
        cv2.line(image,(20,122),(320,122),0,3)
        for x in range(60,300,60): image[120:125,x:x+3]=255
        candidates=detect_dashed_guides(image,tile_size=340)
        guide=next((q for q in candidates if q.get('orientation')=='horizontal' and 40<=q['y1']<=52 and q['classification']=='high-confidence-removable'),None)
        self.assertIsNotNone(guide,candidates)
        mask,protected,classified=build_guide_removal_mask(image,[guide])
        self.assertGreater(cv2.countNonZero(mask),30)
        self.assertEqual(cv2.countNonZero(cv2.bitwise_and(mask,np.where(np.indices(image.shape)[0]==122,255,0).astype(np.uint8))),0)
        cleaned=cv2.inpaint(cv2.cvtColor(image,cv2.COLOR_GRAY2RGB),mask,4,cv2.INPAINT_TELEA)
        self.assertGreater(np.count_nonzero(cleaned[40:53,20:320]!=cv2.cvtColor(image[40:53,20:320],cv2.COLOR_GRAY2RGB)),20)

    def test_diagonal_dash_chain_is_detected_and_boundary_crossing_is_protected(self):
        image=np.full((260,420,3),255,np.uint8)
        cv2.rectangle(image,(10,10),(409,249),(0,0,0),8)
        cv2.line(image,(15,130),(405,130),(0,0,0),3)
        for t in np.arange(0,.98,.055):
            if int(t*20)%2==0:
                x=30+340*t; y=35+170*t
                cv2.line(image,(round(x),round(y)),(round(x+12),round(y+6)),(0,0,0),2)
        detected=detect_dashed_guides(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),tile_size=420)
        self.assertTrue(any(q.get('orientation')=='diagonal' and q.get('classification')=='high-confidence-removable' for q in detected),detected)
        # Include the full chain to validate exact removal with an internal boundary crossing.
        candidate={'x1':30,'y1':35,'x2':370,'y2':205,'classification':'high-confidence-removable','dash_runs':8,'dash_gap_cv':.1}
        mask,protected,classified=build_guide_removal_mask(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),[candidate])
        self.assertGreater(cv2.countNonZero(mask),0)
        self.assertGreater(cv2.countNonZero(protected),0)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected)
        cleaned[protected>0]=image[protected>0]
        self.assertTrue(np.array_equal(image[129:132,190:245],cleaned[129:132,190:245]),'internal boundary pixels changed at guide crossing')
        self.assertTrue(np.array_equal(image[8:19,:,:],cleaned[8:19,:,:]),'thick outer boundary changed')
        self.assertTrue(classified[0]['mask_pixels']>0)
        self.assertTrue(classified[0]['review_required'],'protected crossing must remain a manual-review item')

    def test_edge_adjacent_dash_chain_is_removed_without_weakening_perimeter(self):
        image=np.full((180,320,3),255,np.uint8); cv2.rectangle(image,(8,8),(311,171),(0,0,0),7)
        for x in range(22,298,21): cv2.line(image,(x,25),(x+10,25),(0,0,0),2)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        candidates=detect_dashed_guides(gray,tile_size=320)
        candidate=next((q for q in candidates if q.get('orientation')=='horizontal' and q.get('classification')=='high-confidence-removable' and 22<=q['y1']<=28),None)
        self.assertIsNotNone(candidate,candidates)
        mask,protected,_=build_guide_removal_mask(gray,[candidate])
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected); cleaned[protected>0]=image[protected>0]
        self.assertGreater(np.count_nonzero(np.any(image[22:29,20:300]!=cleaned[22:29,20:300],axis=2)),0)
        self.assertTrue(np.array_equal(image[7:17,:,:],cleaned[7:17,:,:]),'outer edge was weakened by nearby guide cleanup')

    def test_top_and_bottom_dash_trains_remove_only_dash_pixels_near_real_boundaries(self):
        image=np.full((190,340,3),255,np.uint8); cv2.rectangle(image,(8,8),(331,181),(0,0,0),7)
        cv2.line(image,(18,145),(322,145),(0,0,0),3)
        for y in (27,158):
            for x in range(24,305,22): cv2.line(image,(x,y),(x+11,y),(0,0,0),2)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        candidates=[{'x1':24,'y1':27,'x2':310,'y2':27,'classification':'high-confidence-removable','dash_runs':13,'dash_gap_cv':.05},
                    {'x1':24,'y1':158,'x2':310,'y2':158,'classification':'high-confidence-removable','dash_runs':13,'dash_gap_cv':.05}]
        mask,protected,classified=build_guide_removal_mask(gray,candidates,image)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected); cleaned[protected>0]=image[protected>0]
        self.assertGreater(np.count_nonzero(np.any(image[23:32,20:316]!=cleaned[23:32,20:316],axis=2)),50,'top guide remained')
        self.assertGreater(np.count_nonzero(np.any(image[154:163,20:316]!=cleaned[154:163,20:316],axis=2)),30,'bottom guide remained')
        self.assertTrue(np.array_equal(image[141:150,18:323],cleaned[141:150,18:323]),'nearby internal boundary changed')
        self.assertTrue(np.array_equal(image[5:15,:,:],cleaned[5:15,:,:]),'outer top boundary changed')
        self.assertTrue(np.array_equal(image[176:186,:,:],cleaned[176:186,:,:]),'outer bottom boundary changed')
        self.assertTrue(all(item['mask_pixels']>0 for item in classified),classified)

    def test_blue_hatched_symbol_removed_without_touching_boundary_or_label_and_source_unchanged(self):
        image=np.full((220,320,3),255,np.uint8); blue=(30,70,230)
        cv2.rectangle(image,(120,70),(180,130),blue,3)
        for y in range(80,125,8): cv2.line(image,(127,y),(173,y),blue,2)
        cv2.line(image,(20,100),(300,100),(0,0,0),3)
        cv2.putText(image,'15',(220,82),cv2.FONT_HERSHEY_SIMPLEX,1,blue,2,cv2.LINE_AA)
        with TemporaryDirectory() as directory:
            source=Path(directory)/'source.tif'; Image.fromarray(image).save(source)
            original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
            candidates=detect_box_symbols(image,tile_size=320)
            symbol=next((b for b in candidates if b['classification']=='high-confidence-removable' and b['kind']=='blue-hatched-symbol'),None)
            self.assertIsNotNone(symbol,candidates)
            mask,classified=build_blue_symbol_mask(image,[symbol])
            self.assertGreater(cv2.countNonZero(mask),0)
            cleaned=apply_removal_mask(image,mask,dilation=0)
            self.assertGreater(np.count_nonzero(np.any(image[73:127,123:177]!=cleaned[73:127,123:177],axis=2)),100)
            self.assertTrue(np.array_equal(image[98:103,20:300],cleaned[98:103,20:300]),'parcel boundary touching symbol changed')
            self.assertTrue(np.array_equal(image[48:88,215:265],cleaned[48:88,215:265]),'principal number changed')
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),original_hash)
            self.assertTrue(classified[0]['mask_pixels']>0)

    def test_step_shaped_blue_diagram_is_removed_but_boundary_and_nearby_decimal_are_separate(self):
        image=np.full((280,420,3),255,np.uint8); blue=(20,70,220)
        # Concave, internally divided schematic with a small detached mark.
        for p,q in [((120,65),(245,65)),((120,65),(120,155)),((120,155),(195,155)),((195,155),(195,110)),((195,110),(245,110)),((245,65),(245,110))]:
            cv2.line(image,p,q,blue,3)
        for y in range(76,101,5): cv2.line(image,(132,y),(232,y),blue,1)
        cv2.circle(image,(252,112),2,blue,-1)
        cv2.putText(image,'6.6',(165,195),cv2.FONT_HERSHEY_SIMPLEX,.9,blue,2,cv2.LINE_AA)
        cv2.putText(image,'15',(320,145),cv2.FONT_HERSHEY_SIMPLEX,1,blue,2,cv2.LINE_AA)
        cv2.line(image,(20,156),(400,156),(0,0,0),3)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        protected=np.where((gray<180)&((image.max(axis=2).astype(np.int16)-image.min(axis=2).astype(np.int16))<38),255,0).astype(np.uint8)
        candidates=detect_box_symbols(image,tile_size=500)
        symbol=next((b for b in candidates if b['classification']=='high-confidence-removable' and b['x']<130 and b['width']>100),None)
        self.assertIsNotNone(symbol,candidates)
        mask,classified=build_blue_symbol_mask(image,[symbol],protected)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected)
        cleaned[protected>0]=image[protected>0]
        self.assertGreater(cv2.countNonZero(mask),500)
        self.assertTrue(np.array_equal(image[154:159,:],cleaned[154:159,:]))
        self.assertTrue(np.array_equal(image[116:155,310:365],cleaned[116:155,310:365]),'principal label changed')
        self.assertTrue(np.array_equal(image[170:202,155:220],cleaned[170:202,155:220]),'detached measurement was included in the symbol mask')
        self.assertGreater(classified[0]['mask_pixels'],0)
        measurements=detect_color_measurements(image,tile_size=500)+detect_oriented_measurements(image,tile_size=500)
        measurement=next((b for b in measurements if b['text']=='6.6'),None)
        self.assertIsNotNone(measurement,measurements)
        mmask=np.zeros(image.shape[:2],np.uint8)
        for part in measurement.get('component_boxes') or [measurement]:
            x,y,w,h=[part[k] for k in ('x','y','width','height')]
            cv2.rectangle(mmask,(max(0,x-2),max(0,y-2)),(min(image.shape[1]-1,x+w+2),min(image.shape[0]-1,y+h+2)),255,-1)
        final=apply_removal_mask(cleaned,mmask,dilation=1,protected=protected); final[protected>0]=image[protected>0]
        self.assertGreater(np.count_nonzero(np.any(cleaned[170:202,155:220]!=final[170:202,155:220],axis=2)),10)
        self.assertTrue(np.array_equal(image[154:159,:],final[154:159,:]))
        self.assertTrue(np.array_equal(image[116:155,310:365],final[116:155,310:365]))

    def test_detected_diagonal_dash_train_changes_pixels_and_preserves_crossing_boundary(self):
        image=np.full((300,500,3),255,np.uint8); cv2.line(image,(20,170),(480,170),(0,0,0),4)
        for x in range(30,450,28):
            y=35+(x-30)//3
            cv2.line(image,(x,y),(x+11,y+4),(0,0,0),2)
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        candidates=detect_dashed_guides(gray,tile_size=520)
        candidate=next((q for q in candidates if q['orientation']=='diagonal' and q['classification']=='high-confidence-removable'),None)
        self.assertIsNotNone(candidate,candidates)
        mask,protected,classified=build_guide_removal_mask(gray,[candidate],image)
        cleaned=apply_removal_mask(image,mask,dilation=0,protected=protected); cleaned[protected>0]=image[protected>0]
        self.assertGreater(np.count_nonzero(np.any(image!=cleaned,axis=2)),80)
        self.assertTrue(np.array_equal(image[167:174,:],cleaned[167:174,:]),'crossing parcel boundary changed')
        self.assertTrue(classified[0]['mask_pixels']>0)

if __name__=='__main__': unittest.main()

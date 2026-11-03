import argparse
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.ultralytics'))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',required=True)
    p.add_argument('--source',required=True,help='Image, directory, or video path')
    p.add_argument('--conf',type=float,default=0.25)
    a=p.parse_args()
    from ultralytics import YOLO
    model=YOLO(a.weights)
    output=None
    try:
        for frame,r in enumerate(model.predict(source=a.source,conf=a.conf,device=0,stream=True,save=True,project=str(ROOT/'runs/predictions'))):
            if output is None:
                output=(Path(r.save_dir)/'detections.jsonl').open('w',encoding='utf-8')
            boxes=[]
            for b in r.boxes:
                x,y,w,h=b.xywhn[0].tolist()
                boxes.append(dict(class_name=r.names[int(b.cls.item())],confidence=float(b.conf.item()),xyxy=b.xyxy[0].tolist(),image_relative_box_area=w*h,severity='not_assessed'))
            output.write(json.dumps(dict(source=r.path,frame_index=frame,detections=boxes))+'\n')
    finally:
        if output:
            output.close()

if __name__ == '__main__':
    main()

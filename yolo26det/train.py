from ultralytics import YOLO
import os
import wandb

wandb.login(key="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")

# Load a model
model = YOLO("/home/mohamed.abouelhadid/cv703/runs/detect/cv703/baseline_detector_augs_e200/weights/last.pt")
project = "cv703"
model.train(resume=True,data='data_det.yaml',epochs=100,batch=8,imgsz=1280,save_period=25,project=project,name="baseline_detector_augs_e200",
            cos_lr=True,lr0=0.0001,lrf=0.01,warmup_epochs=5,mosaic=1,close_mosaic=10,degrees=5,shear=1,fliplr=0,optimizer='MuSGD',device=0,workers=8)
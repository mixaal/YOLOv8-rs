#!/bin/bash -xe
rm -rf ultra
virtualenv ultra
. ultra/bin/activate
pip3 install ultralytics
yolo export model=yolov8n.pt format=torchscript
yolo export model=yolov8s.pt format=torchscript
yolo export model=yolov8m.pt format=torchscript
yolo export model=yolov8l.pt format=torchscript
yolo export model=yolov8x.pt format=torchscript
yolo export  model=yolov8n-seg.pt format=torchscript
yolo export  model=yolov8n-cls.pt format=torchscript
yolo export  model=yolov8s-cls.pt format=torchscript
yolo export  model=yolov8m-cls.pt format=torchscript
yolo export  model=yolov8l-cls.pt format=torchscript
yolo export  model=yolov8x-cls.pt format=torchscript
ls -l *torchscript
rm -f yolov8n.pt
rm -rf ultra

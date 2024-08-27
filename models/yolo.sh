#!/bin/bash -xe
virtualenv ultra
. ultra/bin/activate
pip3 install ultralytics
yolo export model=yolov8n.pt
ls -l *torchscript
rm -f yolov8n.pt
rm -rf ultra

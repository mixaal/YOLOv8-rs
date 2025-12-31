#!/bin/bash -xe

echo "[💾] Installing virtualenv ..."
sudo apt install python3-virtualenv


echo "[🗑]Removing existing environment ultra..."
rm -rf ultra


echo "[⚡] Creating environment..."
virtualenv ultra
echo "[⚡] Activating environment..."
. ultra/bin/activate
echo "[💾] Installing ultralytics..."
pip3 install ultralytics
echo "[💾] Exporting yolov8n..."
yolo export model=yolov8n.pt format=torchscript
echo "[💾] Exporting yolov8s..."
yolo export model=yolov8s.pt format=torchscript
echo "[💾] Exporting yolov8m..."
yolo export model=yolov8m.pt format=torchscript
echo "[💾] Exporting yolov8l..."
yolo export model=yolov8l.pt format=torchscript
echo "[💾] Exporting yolov8x..."
yolo export model=yolov8x.pt format=torchscript
echo "[💾] Exporting yolov8n-seg..."
yolo export  model=yolov8n-seg.pt format=torchscript
echo "[💾] Exporting yolov8n-cls..."
yolo export  model=yolov8n-cls.pt format=torchscript
echo "[💾] Exporting yolov8s-cls..."
yolo export  model=yolov8s-cls.pt format=torchscript
echo "[💾] Exporting yolov8m-cls..."
yolo export  model=yolov8m-cls.pt format=torchscript
echo "[💾] Exporting yolov8l-cls..."
yolo export  model=yolov8l-cls.pt format=torchscript
echo "[💾] Exporting yolov8x-cls..."
yolo export  model=yolov8x-cls.pt format=torchscript
ls -l *torchscript
echo "[🗑] Cleaning up ..."
rm -f yolov8n.pt
rm -rf ultra

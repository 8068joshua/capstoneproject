# pure_yolov5.py에 이어쓰세요
# trained weights are saved by default in our weights folder
%ls runs/
%ls runs/train/yolov5_results/weights
%cd /content/yolov5/

from google.colab import files
files.download('/content/yolov5/runs/train/yolov5_results/weights/best.pt') 

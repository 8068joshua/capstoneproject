# create the model & config file exporting folder
%mkdir /content/exported_files 
INFO_FILE = "/content/exported_files/info.txt"
!touch $INFO_FILE
!echo "YOLOv5 version:" $VERSION >> $INFO_FILE

# clone YOLOv5 repository
%cd /content
!echo "git clone https://github.com/ultralytics/yolov5" # check the cloning script
!git clone https://github.com/ultralytics/yolov5  # clone
%cd /content/yolov5
!git log -n 1 # check the latest commit

%cd /content/yolov5

# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
!pip install -q roboflow
import torch

from IPython.display import Image, clear_output  # to display images

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

%cd /content/yolov5
#after following the link above, recieve python code with these fields filled in
from roboflow import Roboflow
rf = Roboflow(api_key="oBsY5hFzVYgIUycM9wqA")
project = rf.workspace("scilab").project("ours-fall-detection")
dataset = project.version(3).download("yolov5")

# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
%cat {dataset.location}/data.yaml

# define the epoch number, image size & prefered model file for training
EPOCHS = 15
IMG_SIZE = 416
MODEL = "yolov5l.pt"
MODEL_CONF = "/content/yolov5/models/yolov5l.yaml"

!echo "Epoch:" $EPOCHS >> $INFO_FILE
!echo "Image size:" $IMG_SIZE >> $INFO_FILE
!echo "Base model:" $MODEL >> $INFO_FILE
!echo "Base model config file:" $MODEL_CONF >> $INFO_FILE

#this is the model configuration we will use for our tutorial 
%cat $MODEL_CONF

# copy the custom YAML file
%cp $MODEL_CONF /content/yolov5/models/custom_model.yaml

# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])


with open('/content/yolov5/models/custom_model.yaml' , 'r') as f:
    
    #read file
    file_source = f.read()
    
    #replace 'nc:' with the num_classes in the file
    new_string = 'nc: '+str(num_classes)+' #'
    replace_string = file_source.replace('nc:', new_string)
    
with open('/content/yolov5/models/custom_model.yaml', 'w') as f:
    #save output
    f.write(replace_string)

%cat /content/yolov5/models/custom_model.yaml

# copy the config file to the "exported_files" folder
%cp /content/yolov5/models/custom_model.yaml /content/exported_files
%cp {dataset.location}/data.yaml /content/exported_files

# train the "MODEL" on custom data for "EPOCHS" epochs
# you can increase-decrease the batch size
# time its performance
%%time
%cd /content/yolov5/
!python train.py --img $IMG_SIZE --batch 16 --epochs $EPOCHS --data {dataset.location}/data.yaml --cfg ./models/custom_model.yaml --weights $MODEL --name yolov5_results  --cache

# copy the best model to the "exported_files" folder
%cp /content/yolov5/runs/train/yolov5_results/weights/best.pt /content/exported_files

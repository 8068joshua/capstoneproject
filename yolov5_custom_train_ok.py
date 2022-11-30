# -*- coding: utf-8 -*-
"""YOLOv5_Custom_Train_OK.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15QfPY43XFDgpdB90zgt8wj56y78WLrNe

# How to Train YOLOv5 on Custom Objects

This tutorial is based on the [YOLOv5 repository](https://github.com/ultralytics/yolov5) by [Ultralytics](https://www.ultralytics.com/) and [YOLOv5 training blog post](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/) by [Roboflow](https://roboflow.com). This notebook shows training on your own custom objects. 

You can follow this notebook while reading the blog post on [how to train YOLOv5](https://www.forecr.io/blogs/ai-algorithms/how-to-train-a-yolov5-object-detection-model-in-google-colab), concurrently.

### Steps Covered in this Tutorial

* Install YOLOv5 dependencies
* Download custom YOLOv5 object detection data from Roboflow
* Write the YOLOv5 training configuration
* Run YOLOv5 training
* Evaluate YOLOv5 performance
* Visualize YOLOv5 training data
* Run YOLOv5 inference on test images
* Export saved YOLOv5 weights for future inference

#Install Dependencies

_(Remember to choose GPU in Runtime if not already selected. Runtime --> Change Runtime Type --> Hardware accelerator --> GPU)_
"""

# Commented out IPython magic to ensure Python compatibility.
# create the model & config file exporting folder
# %mkdir /content/exported_files 
INFO_FILE = "/content/exported_files/info.txt"
!touch $INFO_FILE
!echo "YOLOv5 version:" $VERSION >> $INFO_FILE

# Commented out IPython magic to ensure Python compatibility.
# clone YOLOv5 repository
# %cd /content
!echo "git clone https://github.com/ultralytics/yolov5" # check the cloning script
!git clone https://github.com/ultralytics/yolov5  # clone
# %cd /content/yolov5
!git log -n 1 # check the latest commit

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5

# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
!pip install -q roboflow
import torch

from IPython.display import Image, clear_output  # to display images

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

"""# Download Correctly Formatted Custom Dataset 

We'll download our dataset from Roboflow. Use the "**YOLOv5 PyTorch**" export format. Note that the Ultralytics implementation calls for a YAML file defining where your training and test data is. The Roboflow export also writes this format for us.

To get your data into Roboflow, follow the [Getting Started Guide](https://blog.roboflow.ai/getting-started-with-roboflow/).

![YOLOv5 PyTorch export](https://i.imgur.com/5vr9G2u.png)
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5
#after following the link above, recieve python code with these fields filled in
from roboflow import Roboflow
rf = Roboflow(api_key="oBsY5hFzVYgIUycM9wqA")
project = rf.workspace("scilab").project("ours-fall-detection")
dataset = project.version(3).download("yolov5")

# Commented out IPython magic to ensure Python compatibility.
# this is the YAML file Roboflow wrote for us that we're loading into this notebook with our data
# %cat {dataset.location}/data.yaml

# define the epoch number, image size & prefered model file for training
EPOCHS = 15
IMG_SIZE = 416
MODEL = "yolov5l.pt"
MODEL_CONF = "/content/yolov5/models/yolov5l.yaml"

!echo "Epoch:" $EPOCHS >> $INFO_FILE
!echo "Image size:" $IMG_SIZE >> $INFO_FILE
!echo "Base model:" $MODEL >> $INFO_FILE
!echo "Base model config file:" $MODEL_CONF >> $INFO_FILE

"""# Define Model Configuration and Architecture

We will write a yaml script that defines the parameters for our model like the number of classes, anchors, and each layer.

You do not need to edit these cells, but you may.
"""

# Commented out IPython magic to ensure Python compatibility.
#this is the model configuration we will use for our tutorial 
# %cat $MODEL_CONF

# Commented out IPython magic to ensure Python compatibility.
# copy the custom YAML file
# %cp $MODEL_CONF /content/yolov5/models/custom_model.yaml


# define number of classes based on YAML
import yaml
with open(dataset.location + "/data.yaml", 'r') as stream:
    num_classes = str(yaml.safe_load(stream)['nc'])

# Commented out IPython magic to ensure Python compatibility.

with open('/content/yolov5/models/custom_model.yaml' , 'r') as f:
    
    #read file
    file_source = f.read()
    
    #replace 'nc:' with the num_classes in the file
    new_string = 'nc: '+str(num_classes)+' #'
    replace_string = file_source.replace('nc:', new_string)
    
with open('/content/yolov5/models/custom_model.yaml', 'w') as f:
    #save output
    f.write(replace_string)

# %cat /content/yolov5/models/custom_model.yaml

# Commented out IPython magic to ensure Python compatibility.
# copy the config file to the "exported_files" folder
# %cp /content/yolov5/models/custom_model.yaml /content/exported_files
# %cp {dataset.location}/data.yaml /content/exported_files

"""# Train Custom YOLOv5 Detector

### Next, we'll fire off training!


Here, we are able to pass a number of arguments:
- **img:** define input image size
- **batch:** determine batch size
- **epochs:** define the number of training epochs. (Note: often, 3000+ are common here!)
- **data:** set the path to our yaml file
- **cfg:** specify our model configuration
- **weights:** specify a custom path to weights.
- **name:** result names
- **nosave:** only save the final checkpoint
- **cache:** cache images for faster training
"""

# Commented out IPython magic to ensure Python compatibility.
# # train the "MODEL" on custom data for "EPOCHS" epochs
# # you can increase-decrease the batch size
# # time its performance
# %%time
# %cd /content/yolov5/
# !python train.py --img $IMG_SIZE --batch 16 --epochs $EPOCHS --data {dataset.location}/data.yaml --cfg ./models/custom_model.yaml --weights $MODEL --name yolov5_results  --cache
# 
# # copy the best model to the "exported_files" folder
# %cp /content/yolov5/runs/train/yolov5_results/weights/best.pt /content/exported_files

"""# Evaluate Custom YOLOv5 Detector Performance

Training losses and performance metrics are saved to Tensorboard and also to a logfile defined above with the **--name** flag when we train. In our case, we named this `yolov5_results`. (If given no name, it defaults to `results.txt`.) The results file is plotted as a png after training completes.

Note from Glenn: Partially completed `results.txt` files can be plotted with `from utils.utils import plot_results; plot_results()`.
"""

# Commented out IPython magic to ensure Python compatibility.
# Start tensorboard
# Launch after you have started training
# logs save in the folder "runs"
# %load_ext tensorboard
# %tensorboard --logdir runs

# Commented out IPython magic to ensure Python compatibility.
# we can also output some older school graphs if the tensor board isn't working for whatever reason... 
# %cd /content/yolov5/
from utils.plots import plot_results  # plot results.txt as results.png
Image(filename='/content/yolov5/runs/train/yolov5_results/results.png', width=1000)  # view results.png

"""### Visualize Our Training Data with Labels

After training starts, view `train*.jpg` images to see training images, labels and augmentation effects.

Note a mosaic dataloader is used for training (shown below), a new dataloading concept developed by Glenn Jocher and first featured in [YOLOv4](https://arxiv.org/abs/2004.10934).
"""

# first, display our ground truth data
print("GROUND TRUTH TRAINING DATA:")

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/train/yolov5_results/*_batch0_labels.jpg'): #assuming JPG
    display(Image(filename=imageName, width=900))
    break

# print out an augmented training example
print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5_results/train_batch0.jpg', width=900)

"""#Run Inference  With Trained Weights
Run inference with a pretrained checkpoint on contents of `test/images` folder downloaded from Roboflow.
"""

# Commented out IPython magic to ensure Python compatibility.
# trained weights are saved by default in our weights folder
# %ls runs/

# Commented out IPython magic to ensure Python compatibility.
# %ls runs/train/yolov5_results/weights

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5/
!python detect.py --weights runs/train/yolov5_results/weights/best.pt --img $IMG_SIZE --conf 0.4 --source {dataset.location}/test/images

#display inference on ALL test images
#this looks much better with longer training above

import glob
from IPython.display import Image, display

for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'): #assuming JPG
    display(Image(filename=imageName))
    print("\n")

"""# Export Trained Weights for Future Inference

Now that you have trained your custom detector, you can export the trained weights you have made here for inference on your device elsewhere
"""

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cp /content/exported_files/* /content/gdrive/My\ Drive
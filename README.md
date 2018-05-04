# ComputerVisionProject


# Requirements
  * python2
  * python3
  * tensorflow >= 1.1
  * conda
  * pytorch 0.2 [w/ torchvision]
  * cudnn
  * cuda
  * Download ImagetNet weights: https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM
  
  
# Description of repo
This repo is an adaption of the one found here: https://github.com/ruotianluo/ImageCaptioning.pytorch for use with Flickr8k.
  The model folder contains Show and Tell / Show, Attend, and Tell / Top Down Bottom Up and Attend models. 
  To run with flickr8k dataset, please download the images and place them along with the captions in the /data directory.
  Please consult this repository https://github.com/eriche2016/image_caption_with_semantic_attenion/tree/master/flickr8k-caption/annotations for obtaining the Flickr data as json. 
  then run the version of convert_coco_as_raw.py provided in this repo on this file.
  
  After obtaining all required files, run command0 then command2. 
  These commands will prepare the data set so they do not need to be re-processed for different models. 
  
  Then to train a model execute the run_train command. This command can be edited to run the model desired from the models folder. 
  This command will generate the output models in /log_st, which will be over-written by subsequent runs of this command.
  
  In order to evaluate the results, execute run_eval to test the models performance against various metrics.
  
  ***Of note, if any import errors occur, please execute a command similar to this one:
    ``` export PYTHONPATH /some/dir/here ```***
    

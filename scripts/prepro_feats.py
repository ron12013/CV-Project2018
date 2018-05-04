from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import h5py
from random import shuffle, seed

import numpy as np
import torch
from torch.autograd import Variable
import skimage.io

from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from misc.resnet_utils import myResnet
import misc.resnet as resnet


def main(params):
  net = getattr(resnet, params['model'])()
  net.load_state_dict(torch.load(os.path.join(params['model_root'],params['model']+'.pth')))
  my_resnet = myResnet(net)
  my_resnet.cuda()
  my_resnet.eval()

  imgs = json.load(open(params['input_json'], 'r'))
  imgs = imgs['images']
  N = len(imgs)

  seed(123) # make reproducible

  dir_fc = params['output_dir']+'_fc'
  dir_att = params['output_dir']+'_att'
  if not os.path.isdir(dir_fc):
    os.mkdir(dir_fc)
  if not os.path.isdir(dir_att):
    os.mkdir(dir_att)

  with h5py.File(os.path.join(dir_fc, 'feats_fc.h5')) as file_fc,\
       h5py.File(os.path.join(dir_att, 'feats_att.h5')) as file_att:
    for i, img in enumerate(imgs):
      # load the image
      I = skimage.io.imread(os.path.join(params['images_root'], img['split'], img['filename']))
      # handle grayscale input images
      if len(I.shape) == 2:
        I = I[:,:,np.newaxis]
        I = np.concatenate((I,I,I), axis=2)

      I = I.astype('float32')/255.0
      I = torch.from_numpy(I.transpose([2,0,1])).cuda()
      I = Variable(preprocess(I), volatile=True)
      tmp_fc, tmp_att = my_resnet(I, params['att_size'])
      # write to hdf5

      d_set_fc = file_fc.create_dataset(str(img['filename']), 
        (2048,), dtype="float")
      d_set_att = file_att.create_dataset(str(img['filename']), 
        (params['att_size'], params['att_size'], 2048), dtype="float")

      d_set_fc[...] = tmp_fc.data.cpu().float().numpy()
      d_set_att[...] = tmp_att.data.cpu().float().numpy()
      if i % 1000 == 0:
        print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0 / N))
    file_fc.close()
    file_att.close()


if __name__ == "__main__":

  parser = argparse.ArgumentParser()

  # input json
  parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
  parser.add_argument('--output_dir', default='data', help='output directory')

  # options
  parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')
  parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
  parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
  parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict
  print('parsed input parameters:')
  print(json.dumps(params, indent = 2))
  main(params)

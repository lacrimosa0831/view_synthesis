import torch
import torch.nn as nn

#import visdom

from view_syn_trainer import Trainer
from network_syn_mapped import *
from dataset_syn import *
from util import mkdirs, set_caffe_param_mult
import weight_init
import os.path as osp

# --------------
# PARAMETERS
# --------------
network_type = 'SphericalUnet' # 'RectNet' or 'UResNet' or 'SphericalUnet'
experiment_name = 'view_syn_mapped'
input_dir = './data/Realistic' # Dataset location
train_file_list = 'train_tmp.txt' # File with list of training files
val_file_list = 'test_tmp.txt' # File with list of validation files
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = None
#checkpoint_path = osp.join(checkpoint_dir, 'checkpoint_latest.pth')
load_weights_only = False
batch_size = 8
num_workers = 4
lr = 2e-4
step_size = 5
lr_decay = 0.5
num_epochs = 20
validation_freq = 1
visualization_freq = 50
validation_sample_freq = -1
#device_ids = [0,1,2,3]
device_ids = [0]


# -------------------------------------------------------
# Fill in the rest
#vis = visdom.Visdom()
env = experiment_name
device = torch.device('cuda', device_ids[0])

# UResNet

if network_type == 'UResNet':
	model = UResNet()
# RectNet
elif network_type == 'RectNet':
	model = RectNet()
# 	SphericalUnet
elif network_type == 'SphericalUnet':
	model = SphericalUnet()
	#weight_init.initialize_weights(model, init="xavier", pred_bias=float(5.0))
    
else:
	assert False, 'Unsupported network type'

result_view_dir = 'view_syn_results_mapped'

# Make the checkpoint directory
mkdirs(checkpoint_dir)
mkdirs(result_view_dir)
num_param = sum([x.nelement() for x in model.parameters() if x.requires_grad])
print('## batch size: ', batch_size)
print('## learning rate: ', lr)
print('## classifer parameters:', num_param)

# -------------------------------------------------------
# Set up the training routine
network = nn.DataParallel(
	model.float(),
	device_ids=device_ids).to(device)

train_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=train_file_list),
	batch_size=batch_size,
	shuffle=True,
	num_workers=num_workers,
	drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list),
	batch_size=batch_size,
	shuffle=False,
	num_workers=num_workers,
	drop_last=True)


# Set up network parameters with Caffe-like LR multipliers
param_list = set_caffe_param_mult(network, lr, 0)
optimizer = torch.optim.Adam(
	params=model.parameters(),
	betas=(0.9,0.999),
	eps=1e-8,
	lr=lr,
	weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
	step_size=step_size,
	gamma=lr_decay)

trainer = Trainer(
	experiment_name, 
	network, 
	train_dataloader, 
	val_dataloader, 
	optimizer,
	checkpoint_dir,
	device,
	result_view_dir=result_view_dir,
	visdom=None,
	scheduler=scheduler, 
	num_epochs=num_epochs,
	validation_freq=validation_freq,
	visualization_freq=visualization_freq, 
	validation_sample_freq=validation_sample_freq)



trainer.train(checkpoint_path, load_weights_only)

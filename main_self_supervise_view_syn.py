import torch
import torch.nn as nn

#import visdom

from view_syn_disparity import Trainer
from network_syn import *
from dataset_syn import *
from util import mkdirs, set_caffe_param_mult
from models import *
import os.path as osp

# --------------
# PARAMETERS
# --------------
network_type = 'SphericalUnet' # 'RectNet' or 'UResNet' or 'SphericalUnet'
experiment_name = 'view_syn3'
input_dir = './data/Realistic' # Dataset location
train_file_list = 'part_train.txt' # File with list of training files
val_file_list = 'test_tmp.txt' # File with list of validation files
checkpoint_dir = osp.join('experiments', experiment_name)
checkpoint_path = None
#checkpoint_path = osp.join(checkpoint_dir, 'checkpoint_latest.pth')
load_weights_only = False
batch_size = 2
num_workers = 4
lr = 2e-4
step_size = 5
lr_decay = 0.5
num_epochs = 50
validation_freq = 1
visualization_freq = 20
validation_sample_freq = -1
#device_ids = [0,1,2,3]
device_ids = [0, 1]


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

else:
	assert False, 'Unsupported network type'

result_view_dir = 'view_syn_disparity_results'

# Make the checkpoint directory
mkdirs(checkpoint_dir)
mkdirs(result_view_dir)
num_param = sum([x.nelement() for x in model.parameters() if x.requires_grad])
print('## batch size: ', batch_size)
print('## learning rate: ', lr)
print('## classifer parameters:', num_param)

# -------------------------------------------------------
# Set up the training routine
network1 = nn.DataParallel(
	model.float(),
	device_ids=device_ids).to(device)

disparity_model = LCV_ours_sub3(68)
init_array = np.zeros((1,1, 7, 1))	# 7 of filter
init_array[:,:,3,:] = 28./540
init_array[:,:,2,:] = 512./540
disparity_model.forF.forfilter1.weight = torch.nn.Parameter( torch.Tensor( init_array))
network2 = nn.DataParallel(
	disparity_model.float(),
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
param_list1 = set_caffe_param_mult(network1, lr, 0)
param_list2 = set_caffe_param_mult(network2, lr, 0)
optimizer1 = torch.optim.Adam(
	params=param_list1,
	lr=lr)

optimizer2 = torch.optim.Adam(
	params=param_list2,
	lr=1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, 
	step_size=step_size,
	gamma=lr_decay)

trainer = Trainer(
	experiment_name, 
	network1, 
    network2,
	train_dataloader, 
	val_dataloader, 
	optimizer1,
    optimizer2,
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

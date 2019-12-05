from __future__ import print_function
import argparse
import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
from torch.utils.tensorboard import SummaryWriter
from skimage.measure import compare_mse as mse
from tqdm import tqdm
from dataset_syn import OmniDepthDataset
import cv2
import spherical as S360

parser = argparse.ArgumentParser(description='360SD-Net')
parser.add_argument('--maxdisp', type=int ,default=68,
                    help='maxium disparity')
parser.add_argument('--model', default='360SDNet',
                    help='select model')
parser.add_argument('--datapath', default='data/MP3D/train/',
                    help='datapath')
parser.add_argument('--datapath_val', default='data/MP3D/val/',
                    help='datapath for validation')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--start_decay', type=int, default=400,
                    help='number of epoch for lr to start decay')
parser.add_argument('--start_learn', type=int, default=40,
                    help='number of epoch for LCV to start learn')
parser.add_argument('--batch', type=int, default=12,
                    help='number of batch to train')
parser.add_argument('--checkpoint', default= None,
                    help='load checkpoint path')
parser.add_argument('--save_checkpoint', default='./checkpoints',
                    help='save checkpoint path')
parser.add_argument('--tensorboard_path', default='./logs',
                    help='tensorboard path')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--real', action='store_true', default=False,
                    help='adapt to real world images in both training and validation')
parser.add_argument('--SF3D', action='store_true', default=False,
                    help='read stanford3D data')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# tensorboard Path -----------------------
writer_path = args.tensorboard_path
if args.SF3D:
    writer_path += '_SF3D'
if args.real:
    writer_path +='_real'
writer = SummaryWriter(writer_path)

#-----------------------------------------

# Random Seed -----------------------------
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
#------------------------------------------
input_dir = './data/Realistic' # Dataset location
train_file_list = 'train_tmp.txt' # File with list of training files
val_file_list = 'test_tmp.txt' # File with list of validation files

# Create Angle info ------------------------------------------------
# Y angle
angle_y = np.array([(i-0.5)/256*180 for i in range(128, -128, -1)])
angle_ys = np.tile(angle_y[:, np.newaxis, np.newaxis], (1, 512, 1))
equi_info = angle_ys
#-------------------------------------------------------------------
batch_size=args.batch
perturb = 0.22
train_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=train_file_list,
		perturb = perturb),
	batch_size=batch_size,
	shuffle=True,
	num_workers=8,
	drop_last=True)

val_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list,
		perturb = perturb),
	batch_size=batch_size,
	shuffle=False,
	num_workers=4,
	drop_last=True)


# Load model ----------------------------------------------
if args.model == '360SDNet':
    model = LCV_ours_sub3(args.maxdisp)
else:
    print('Model Not Implemented!!!')
#----------------------------------------------------------

# assign initial value of filter cost volume ---------------------------------
init_array = np.zeros((1,1, 7, 1))	# 7 of filter
init_array[:,:,3,:] = 28./540
init_array[:,:,2,:] = 512./540
model.forF.forfilter1.weight = torch.nn.Parameter( torch.Tensor( init_array))
#-----------------------------------------------------------------------------

# Multi_GPU for model ----------------------------
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()
#-------------------------------------------------

# Load Checkpoint -------------------------------
start_epoch = 0
if args.checkpoint is not None:
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict['state_dict'])
    start_epoch = state_dict['epoch']
    # load pretrain from MP3D for SF3D
    if start_epoch == 50 and args.SF3D:
        start_epoch=0
        print("MP3D pretrained 50 epoch for SF3D Loaded!!!")
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
#--------------------------------------------------

# Optimizer ----------
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))
#---------------------

# Freeze Unfreeze Function 
# freeze_layer ----------------------
def freeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = False
freeze_layer(model.module.forF.forfilter1)	# if use nn.DataParallel(model), model.module.filtercost instead model.filtercost
# Unfreeze_layer --------------------
def unfreeze_layer(layer):
	for param in layer.parameters():
		param.requires_grad = True
#------------------------------------
result_view_dir = 'view_disparity'
if not os.path.exists(result_view_dir):
    os.makedirs(result_view_dir)   

# Train Function -------------------
def train(imgU, imgD, disp, mask, batch_idx):
    model.train()
    imgU = Variable(torch.FloatTensor(imgU.float()))
    imgD = Variable(torch.FloatTensor(imgD.float()))   
    disp = Variable(torch.FloatTensor(disp.float()))
    mask = mask>0
    # cuda?
    if args.cuda:
        imgU, imgD, disp_true, mask = imgU.cuda(), imgD.cuda(), disp.cuda(), mask.cuda()

    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask.detach_()

    optimizer.zero_grad()
    # Loss -------------------------------------------- 
    output1, output2, output3 = model(imgU,imgD)
    output1 = torch.squeeze(output1,1)
    output2 = torch.squeeze(output2,1)
    output3 = torch.squeeze(output3,1)
    loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
    #--------------------------------------------------
    gt = imgU[:,:3,:,:].detach().cpu().numpy()
    render = imgD[:,:3,:,:].detach().cpu().numpy()
    disp_np = disp.detach().cpu().numpy()
    disp_pred_np = output3.detach().cpu().numpy()

    if batch_idx % 50 == 0:
        for i in range(2):
            gt_img = gt[i, :, :, :].transpose(1,2,0)
            render_img = render[i, :, :, :].transpose(1,2,0)
            disp_gt_img = cv2.normalize(disp_np[i,:,:],None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            disp_pred_img = cv2.normalize(disp_pred_np[i,:,:],None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            cv2.imwrite(result_view_dir + '/gt_' + str(i) + '.png', gt_img*255)
            cv2.imwrite(result_view_dir + '/render_' + str(i) + '.png', render_img*255)
            cv2.imwrite(result_view_dir + '/disp_' + str(i) + '.png', disp_gt_img)
            cv2.imwrite(result_view_dir + '/disp_pred_' + str(i) + '.png', disp_pred_img)
    loss.backward()
    optimizer.step()

    return loss.item()

# Valid Function -----------------------
def val(imgU,imgD,disp_true,mask,batch_idx):
    model.eval()
    imgU   = Variable(torch.FloatTensor(imgU.float()))
    imgD   = Variable(torch.FloatTensor(imgD.float()))   
    # cuda?
    if args.cuda:
        imgU, imgD = imgU.cuda(), imgD.cuda()
    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = mask>0
    with torch.no_grad():
        _,_,output3 = model(imgU,imgD)
    gt = imgU[:,:3,:,:].detach().cpu().numpy()
    render = imgD[:,:3,:,:].detach().cpu().numpy()
    disp_np = disp_true.detach().cpu().numpy()
    disp_pred_np = output3.detach().cpu().numpy()
    if batch_idx % 50 == 0:
        for i in range(2):
            gt_img = gt[i, :, :, :].transpose(1,2,0)
            render_img = render[i, :, :, :].transpose(1,2,0)
            disp_gt_img = cv2.normalize(disp_np[i,:,:],None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            disp_pred_img = cv2.normalize(disp_pred_np[i,:,:],None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            cv2.imwrite(result_view_dir + '/test_gt_' + str(i) + '.png', gt_img*255)
            cv2.imwrite(result_view_dir + '/test_render_' + str(i) + '.png', render_img*255)
            cv2.imwrite(result_view_dir + '/test_disp_' + str(i) + '.png', disp_gt_img)
            cv2.imwrite(result_view_dir + '/test_disp_pred_' + str(i) + '.png', disp_pred_img)

    output = torch.squeeze(output3.data.cpu(),1)
    if len(disp_true[mask])==0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask]-disp_true[mask]))  # end-point-error

    return loss, output

# Adjust Learning Rate
def adjust_learning_rate(optimizer, epoch):
    
    lr = 0.001
    if epoch > args.start_decay:
        lr = 0.0001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Disparity to Depth Function
def todepth(disp):
    H = 256 # image height
    W = 512 # image width
    b = 0.3 # baseline
    theta_T = math.pi - ((np.arange(H).astype(np.float64) + 0.5) * math.pi/H)
    theta_T = np.tile(theta_T[:,None], (1,W))
    angle = b * np.sin(theta_T)
    angle2 = b * np.cos(theta_T)
    #################
    for i in range(len(disp)):
        mask = disp[i,:,:] ==0
        mask_n0 = disp[i,:,:] !=0
        de = np.zeros(disp.shape)
        de[i,:,:] = angle / np.tan( disp[i,:,:] /180 *math.pi) + angle2
        de[i,:,:][mask] = 0
    return de

# Main Function ---------------------------------------------------------------------------------------------
def main():
    global_step = 0
    global_val = 0

    # Start Training ---------------------------------------------------------
    start_full_time = time.time()
    for epoch in tqdm(range(start_epoch+1, args.epochs+1), desc='Epoch'):
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)

	    # unfreeze filter --------------
        if epoch >= args.start_learn:
            unfreeze_layer(model.module.forF.forfilter1)
        #-------------------------------

        # Train --------------------------------------------------------------------------------------------------
        for batch_idx, (up, down, depth, depth_upmask, depth_downmask) in tqdm(enumerate(train_dataloader), desc='Train iter'):
            sgrid = S360.grid.create_spherical_grid(512)
            uvgrid = S360.grid.create_image_grid(512, 256)
            disparity = S360.derivatives.dtheta_vertical(sgrid, depth, perturb) 
            disparity[torch.isnan(disparity)] = 0.0
            disparity[torch.isinf(disparity)] = 0.0
            disparity = disparity.squeeze(1)  
            loss = train(up, down, disparity, depth_upmask, batch_idx)
            total_train_loss += loss
            global_step += 1
            writer.add_scalar('loss',loss,global_step) # tensorboardX for iter
        writer.add_scalar('total train loss',total_train_loss/len(train_dataloader),epoch) # tensorboardX for epoch
        #---------------------------------------------------------------------------------------------------------

        # Save Checkpoint -------------------------------------------------------------
        if not os.path.isdir(args.save_checkpoint):
            os.makedirs(args.save_checkpoint)
        if args.save_checkpoint[-1] == '/':
            args.save_checkpoint = args.save_checkpoint[:-1]
        savefilename = args.save_checkpoint+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(train_dataloader),
            }, savefilename)
        #-----------------------------------------------------------------------------

        # Valid ----------------------------------------------------------------------------------------------------
        total_val_loss = 0
        total_val_crop_rmse = 0
        for batch_idx, (up, down, depth, depth_upmask, depth_downmask) in tqdm(enumerate(val_dataloader), desc='Valid iter'):
            sgrid = S360.grid.create_spherical_grid(512)
            uvgrid = S360.grid.create_image_grid(512, 256)
            disparity = S360.derivatives.dtheta_vertical(sgrid, depth, perturb) 
            disparity[torch.isnan(disparity)] = 0.0
            disparity[torch.isinf(disparity)] = 0.0 
            disparity = disparity.squeeze(1)
            val_loss, val_output = val(up, down, disparity, depth_upmask, batch_idx)

            # for depth cropped rmse -------------------------------------
            depth_gt = todepth( disparity.data.cpu().numpy())
            mask_de_gt = depth_gt >0
            val_crop_rmse = np.sqrt(np.mean(( todepth(val_output.data.cpu().numpy())[mask_de_gt] - depth_gt[mask_de_gt] )**2))
	        #-------------------------------------------------------------
            # Loss ---------------------------------
            total_val_loss += val_loss
            total_val_crop_rmse += val_crop_rmse
            #---------------------------------------
            # Step ------
            global_val+=1
            #------------
        writer.add_scalar('total validation loss',total_val_loss/(len(val_dataloader)),epoch) #tensorboardX for validation in epoch        
        writer.add_scalar('total validation crop 26 depth rmse',total_val_crop_rmse/(len(val_dataloader)),epoch) #tensorboardX rmse for validation in epoch
    writer.close()
    # End Training
    print("Training Ended!!!")
    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))
#----------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
        

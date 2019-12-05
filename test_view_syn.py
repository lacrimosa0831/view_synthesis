import torch
import torch.nn as nn
from torch.autograd import Variable
from view_syn_trainer import Trainer
from network_syn import *
from dataset_syn import *
from util import mkdirs, set_caffe_param_mult
from models import *
import spherical as S360
import supervision as L
import os.path as osp
import util
from tqdm import tqdm
import cv2

# --------------
# PARAMETERS
# --------------
experiment_name = './experiments/view_syn2'
input_dir = './data/Realistic' # Dataset location
train_file_list = 'part_train.txt' # File with list of training files
val_file_list = 'test_tmp.txt' # File with list of validation files
checkpoint_dir = osp.join(experiment_name, 'checkpoint_020.pth')
num_workers = 4
validation_sample_freq = -1
device_ids = [0]

# -------------------------------------------------------
# Fill in the rest
env = experiment_name
device = torch.device('cuda', device_ids[0])

sphereunet = SphericalUnet()
checkpoint1 = torch.load(checkpoint_dir)
util.load_partial_model(sphereunet, checkpoint1['state_dict'])
model = LCV_ours_sub3(68)
checkpoint2 = torch.load('checkpoints/checkpoint_49.tar')
util.load_partial_model(model, checkpoint2['state_dict'])
# -------------------------------------------------------
# Set up the training routine
view_syn_network = sphereunet.float().to(device)

disparity_network = model.float().to(device)

val_dataloader = torch.utils.data.DataLoader(
	dataset=OmniDepthDataset(
		root_path=input_dir, 
		path_to_img_list=val_file_list),
	batch_size=4,
	shuffle=True,
	num_workers=num_workers,
	drop_last=False)

result_view_dir = 'view_disparity_self_supervise_full_data'

def renderfromdisparity(image1, disparity, uvgrid, sgrid):
    disp = torch.cat(
        (
          torch.zeros_like(disparity),
          disparity
        ), dim=1
    )
    render_coords = uvgrid + disp
    render_coords[torch.isnan(render_coords)] = 0
    render_coords[torch.isinf(render_coords)] = 0
    render_rgb, render_mask = L.splatting.render(image1, disparity, render_coords, max_depth=8)
    return render_rgb, render_mask

def val(imgU,imgD,disp_true,mask,batch_idx):
    #imgU   = Variable(torch.FloatTensor(imgU.float()))
    #imgD   = Variable(torch.FloatTensor(imgD.float()))   
    # cuda?
    #if args.cuda:
    imgU, imgD, disp_true = imgU.cuda(), imgD.cuda(), disp_true.cuda()
    # mask value
    #mask = (disp_true < args.maxdisp) & (disp_true > 0)
    mask = mask>0
    with torch.no_grad():
        _,_,output3 = disparity_network(imgU,imgD)
  
    sgrid = S360.grid.create_spherical_grid(512).cuda()
    uvgrid = S360.grid.create_image_grid(512, 256).cuda()    
    gt_render, _ = renderfromdisparity(imgD[:,:3,:,:], disp_true.unsqueeze(1), uvgrid, sgrid)
    render_rgb, _ = renderfromdisparity(imgD[:,:3,:,:], output3.unsqueeze(1), uvgrid, sgrid)
    up = imgU[:,:3,:,:].detach().cpu().numpy()
    down = imgD[:,:3,:,:].detach().cpu().numpy()
    pred_render = render_rgb[:,:3,:,:].detach().cpu().numpy()
    gt_render = gt_render.detach().cpu().numpy()
    disp_np = disp_true.detach().cpu().numpy()
    disp_pred_np = output3.detach().cpu().numpy()

    if batch_idx % 50 == 0:
        for i in range(2):
            up_img = up[i, :3, :, :].transpose(1,2,0)
            down_img = down[i, :3, :, :].transpose(1,2,0)
            pred_render_img = pred_render[i, :, :, :].transpose(1,2,0)
            gt_render_img = gt_render[i, :, :, :].transpose(1,2,0)
            disp_gt_img = cv2.normalize(disp_np[i,:,:],None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            disp_pred_img = cv2.normalize(disp_pred_np[i,:,:],None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

            cv2.imwrite(result_view_dir + '/full_test_up_' + str(i) + '.png', up_img*255)
            cv2.imwrite(result_view_dir + '/full_test_down_' + str(i) + '.png', down_img*255)
            cv2.imwrite(result_view_dir + '/full_test_disp_' + str(i) + '.png', disp_gt_img)
            cv2.imwrite(result_view_dir + '/full_test_disp_pred_' + str(i) + '.png', disp_pred_img)
            cv2.imwrite(result_view_dir + '/full_test_pred_render_' + str(i) + '.png', pred_render_img*255)
            cv2.imwrite(result_view_dir + '/full_test_gt_render_' + str(i) + '.png', gt_render_img*255)

    output = torch.squeeze(output3.data.cpu(),1)
    if len(disp_true[mask])==0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output[mask]-disp_true.cpu()[mask]))  # end-point-error

    return loss, output

view_syn_network.eval()
disparity_network.eval()

with torch.no_grad():
    for batch_idx, (up, down, depth, depth_upmask, depth_downmask) in tqdm(enumerate(val_dataloader), desc='Valid iter'):
            up, down = up.cuda(), down.cuda()
            up_pred = view_syn_network(down[:,:3,:,:])
            print(down[:,:3,:,:].min(), down[:,:3,:,:].max())
            print(up_pred.min(), up_pred.max())
            exit()
            up_pred = torch.cat([up_pred, up[:,3:,:,:]], 1)
            sgrid = S360.grid.create_spherical_grid(512)
            uvgrid = S360.grid.create_image_grid(512, 256)
            disparity = S360.derivatives.dtheta_vertical(sgrid, depth, 0.24) 
            disparity[torch.isnan(disparity)] = 0.0
            disparity[torch.isinf(disparity)] = 0.0 
            disparity = disparity.squeeze(1)
            val_loss, val_output = val(up_pred, down, disparity, depth_upmask, batch_idx)

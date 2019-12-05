import torch
import torch.nn.functional as F
import time

import math
import shutil
import os.path as osp
import scipy.io
import time
import util
import cv2
from vgg_loss import LossNetwork
import torchvision.models.vgg as vgg
import torch.utils.model_zoo as model_zoo
from supervision.photometric import *
import spherical as S360
import supervision as L

vgg_model = vgg.vgg19(pretrained=True)
device_ids = [0, 1]
device = torch.device('cuda', device_ids[0])
vgg_model.to(device)
loss_network = LossNetwork(vgg_model)
loss_network.eval()

def renderfromdisparity(image1, disparity, uvgrid, sgrid):
    disp = torch.cat(
        (
          torch.zeros_like(disparity),
          #disparity
          S360.derivatives.dtheta_vertical(sgrid, disparity, 0.24)
        ), dim=1
    )
    render_coords = uvgrid + disp
    render_coords[torch.isnan(render_coords)] = 0
    render_coords[torch.isinf(render_coords)] = 0
    render_rgb, render_mask = L.splatting.render(image1, disparity, render_coords, max_depth=20)
    return render_rgb, render_mask

photo_params = L.photometric.PhotometricLossParameters(
        alpha=0.85, l1_estimator='none', ssim_estimator='none',
        ssim_mode='gaussian', std=1.5, window=7
    )
def view_syn_loss(image1, image2, disparity, uvgrid, sgrid, mask):
    #disparity = disparity.unsqueeze(1)
    render_rgb, render_mask = renderfromdisparity(image1, disparity, uvgrid, sgrid)
    photo_loss = L.photometric.calculate_loss(render_rgb, image2, photo_params, mask)
    xyz = S360.cartesian.coords_3d(sgrid, disparity)
    dI_dxyz = S360.derivatives.dV_dxyz(xyz)
    duv = S360.derivatives.dI_duv(image1)
    depth_smooth_loss = L.smoothness.guided_smoothness_loss(dI_dxyz, duv, render_mask)
    loss = photo_loss + depth_smooth_loss*0.1
    return render_rgb, loss

# From https://github.com/fyu/drn
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def to_dict(self):
        return {'val' : self.val,
            'sum' : self.sum,
            'count' : self.count,
            'avg' : self.avg}

    def from_dict(self, meter_dict):
        self.val = meter_dict['val']
        self.sum = meter_dict['sum']
        self.count = meter_dict['count']
        self.avg = meter_dict['avg']



def visualize_rgb(rgb):
    # Scale back to [0,255]
    return (255 * rgb).byte()

def visualize_mask(mask):
    '''Visualize the data mask'''
    mask /= mask.max()
    return (255 * mask).byte()


class Trainer(object):

    def __init__(
         self, 
         name, 
         network, 
         train_dataloader, 
         val_dataloader,  
         optimizer,
         checkpoint_dir,
         device, 
         result_view_dir,
         visdom=None,
         scheduler=None, 
         num_epochs=20,
         validation_freq=1,
         visualization_freq=5, 
         validation_sample_freq=-1):

        # Name of this experiment
        self.name = name

        # Class instances
        self.result_view_dir = result_view_dir
        self.network = network
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.photoparams = PhotometricLossParameters(alpha=0.85, l1_estimator='none', ssim_estimator='none',
                        ssim_mode='gaussian', std=1.5, window=7)
        # Training options
        self.num_epochs = num_epochs
        self.validation_freq = validation_freq
        self.visualization_freq = visualization_freq
        self.validation_sample_freq = validation_sample_freq

        # CUDA info
        self.device = device

        # Some timers
        self.batch_time_meter = AverageMeter()
        self.forward_time_meter = AverageMeter()
        self.backward_time_meter = AverageMeter()

        # Some trackers
        self.epoch = 0

        # Directory to store checkpoints
        self.checkpoint_dir = checkpoint_dir

        # Accuracy metric trackers
        self.abs_rel_error_meter = AverageMeter()
        self.sq_rel_error_meter = AverageMeter()
        self.lin_rms_sq_error_meter = AverageMeter()
        self.log_rms_sq_error_meter = AverageMeter()
        self.d1_inlier_meter = AverageMeter()
        self.d2_inlier_meter = AverageMeter()
        self.d3_inlier_meter = AverageMeter()

        # Track the best inlier ratio recorded so far
        self.best_d1_inlier = 0.0
        self.is_best = False

        # List of length 2 [Visdom instance, env]
        self.vis = visdom

        # Loss trackers
        self.loss = AverageMeter()
        

    def forward_pass(self, inputs):
        '''
        Accepts the inputs to the network as a Python list
        Returns the network output
        '''
        return self.network(*inputs)

    def compute_loss(self, render_view, output):
        '''
        Returns the total loss
        '''
        
        with torch.no_grad():
            render_view = render_view.detach()
        
        features_render_view = loss_network(render_view)
        features_output = loss_network(output)

        with torch.no_grad():
            features_render_view = features_render_view[2].detach()
        return calculate_loss(output, render_view, self.photoparams,\
            torch.ones_like(output, dtype=torch.float32, device=self.device))\
                + F.mse_loss(features_output[2], features_render_view)
        
    def backward_pass(self, loss):
        # Computes the backward pass and updates the optimizer
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 3, norm_type='inf')
        self.optimizer.step()

    def train_one_epoch(self):
        print('*************************Train Epoch {}************************'.format(self.epoch+1))
        # Put the model in train mode 
        self.network = self.network.train()

        # Load data
        end = time.time()
        start = end
        train_loss = 0.0

        for batch_num, data in enumerate(self.train_dataloader):

            # Parse the data into inputs, ground truth, and other
            up, down, up_mask = self.parse_data(data)
            
            up = up[:,:3,:,:]
            down = down[:,:3,:,:]
            # Run a forward pass
            forward_time = time.time()
            output = self.forward_pass([down])
            output = torch.abs(output)
            #output = torch.softmax(output, dim=1)
            self.forward_time_meter.update(time.time() - forward_time)
            
            # Compute the loss(es)
            #loss = self.compute_loss(up, output)
            sgrid = S360.grid.create_spherical_grid(512).cuda()
            uvgrid = S360.grid.create_image_grid(512, 256).cuda()
            '''
            rendered_merge = torch.zeros_like(up, dtype=torch.float32)
            mask = torch.zeros([up.shape[0], 1, up.shape[2], up.shape[3]], dtype=torch.int64, device=up.device)  

            for n in range(output.shape[1]):
                layer_pred = output[:,n,:,:].unsqueeze(1)
                render_rgb, render_mask = renderfromdisparity(down, layer_pred, uvgrid, sgrid)
                rendered_merge += render_rgb
                #print(rendered_merge.min().item(), rendered_merge.max().item())
                mask += render_mask
            mask = mask>0
            photo_loss = L.photometric.calculate_loss(rendered_merge, up, photo_params, mask) 
            loss = photo_loss
            #loss = F.l1_loss(rendered_merge, up)
            '''
            rendered_merge, loss = view_syn_loss(down, up, output, uvgrid, sgrid, up_mask)
            # Backpropagation of the total loss
            backward_time = time.time()
            self.backward_pass(loss)
            self.backward_time_meter.update(time.time() - backward_time)

            # Update batch times
            self.batch_time_meter.update(time.time() - end)
            end = time.time()
            train_loss += loss.item()

            # Every few batches
            if batch_num % self.visualization_freq == 0:

                # Visualize the loss
                #self.visualize_loss(batch_num, loss)
                #self.visualize_samples(inputs, gt, other, output)
                
                # Print the most recent batch report
                #self.print_batch_report(batch_num, loss)
                """
                scipy.io.savemat('check.mat', {'gt':inputs[0].detach().cpu().numpy(),
                                           'render_view':render_view.detach().cpu().numpy(),
                                                                                  
                                           'output_view': output.detach().cpu().numpy()})
                """                           
                down_imgs = down.detach().cpu().numpy()
                up_imgs = up.detach().cpu().numpy()
                up_preds = rendered_merge.detach().cpu().numpy()
                depth_preds = output.sum(1).detach().cpu().numpy()
                for i in range(2):
                    d = down_imgs[i, :3, :, :].transpose(1,2,0)
                    u = up_imgs[i, :, :, :].transpose(1,2,0)
                    pred_u = up_preds[i, :, :, :].transpose(1,2,0)
                    pred_u = cv2.normalize(pred_u,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    depth = depth_preds[i, :, :]
                    depth = cv2.normalize(depth,None,255.0,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    cv2.imwrite(self.result_view_dir + '/down_' + str(i) + '.png', d*255)
                    cv2.imwrite(self.result_view_dir + '/up_' + str(i) + '.png', u*255)
                    cv2.imwrite(self.result_view_dir + '/up_pred_' + str(i) + '.png', pred_u)
                    cv2.imwrite(self.result_view_dir + '/depth_pred_' + str(i) + '.png', depth)


                print('Epoch: [{}/{}][{}/{}], '
                      'total loss: {:.4f}'.format(
                       self.epoch+1, 
                       self.num_epochs,
                       batch_num+1,
                       len(self.train_dataloader), 
                       train_loss/(batch_num+1)))
                       
        print('Average loss for epoch {}:\t'
              'Use time: {} s\n'.format(
                  self.epoch+1,
                  time.time()-start) )

    def validate(self):
        print('*************************Validate Epoch {}************************'.format(self.epoch+1))

        # Put the model in eval mode
        self.network = self.network.eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        val_loss = 0.0
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                
                # Parse the data
                up, down, up_mask = self.parse_data(data)
                up = up[:,:3,:,:]
                down = down[:,:3,:,:]
                # Run a forward pass
                output = self.forward_pass([down])
                '''
                output = torch.softmax(output, dim=1)

                sgrid = S360.grid.create_spherical_grid(512).cuda()
                uvgrid = S360.grid.create_image_grid(512, 256).cuda()
                rendered_merge = torch.zeros_like(up, dtype=torch.float32)
                mask = torch.zeros([up.shape[0], 1, up.shape[2], up.shape[3]], dtype=torch.int64, device=up.device)  

                for n in range(output.shape[1]):
                    layer_pred = output[:,n,:,:].unsqueeze(1)
                    render_rgb, render_mask = renderfromdisparity(down, layer_pred, uvgrid, sgrid)
                    rendered_merge += render_rgb
                '''    
                loss = F.mse_loss(output, up)
                down_imgs = down.detach().cpu().numpy()
                up_imgs = up.detach().cpu().numpy()
                up_preds = output.detach().cpu().numpy()
                #depth_preds = output.detach().cpu().numpy()
                
                for i in range(2):
                    d = down_imgs[i, :3, :, :].transpose(1,2,0)
                    u = up_imgs[i, :, :, :].transpose(1,2,0)
                    pred_u = up_preds[i, :, :, :].transpose(1,2,0)
                    #depth = depth_preds[i, 0, :, :]
                    cv2.imwrite(self.result_view_dir + '/test_down_' + str(i) + '.png', d*255)
                    cv2.imwrite(self.result_view_dir + '/test_up_' + str(i) + '.png', u*255)
                    cv2.imwrite(self.result_view_dir + '/test_up_pred_' + str(i) + '.png', pred_u*255)
                    #cv2.imwrite(self.result_view_dir + '/test_depth_pred_' + str(i) + '.png', depth*20)
                val_loss += loss.item()
        val_loss /= batch_num+1

        # Print a report on the validation results
        print('Validation finished in {} seconds'.format(time.time() - s))
        print('Validation loss {}'.format(val_loss))
        #self.print_validation_report()

    def train(self, checkpoint_path=None, weights_only=False):
        print('Starting training')

        # Load pretrained parameters if desired
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, weights_only)
            if weights_only:
                self.initialize_visualizations()
        #else:
            # Initialize any training visualizations
            #self.initialize_visualizations()

        # Train for specified number of epochs
        for self.epoch in range(self.epoch, self.num_epochs):

            # Run an epoch of training
            self.train_one_epoch()
            # Increment the LR scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            if self.epoch % self.validation_freq == 0:
                self.validate()
                self.save_checkpoint()
                #self.visualize_metrics()



    def evaluate(self, checkpoint_path):
        print('Evaluating model....')

        # Load the checkpoint to evaluate
        self.load_checkpoint(checkpoint_path, True, True)

        # Put the model in eval mode
        self.network = self.network.eval()

        # Reset meter
        self.reset_eval_metrics()

        # Load data
        s = time.time()
        evaluate_loss = 0.0
        with torch.no_grad():
            for batch_num, data in enumerate(self.val_dataloader):
                
                # Parse the data
                up, down = self.parse_data(data)
                up = up[:,:3,:,:]
                down = down[:,:3,:,:]

                # Run a forward pass
                output = self.forward_pass(inputs)

                loss = F.mse_loss(up, output)
                down_imgs = down.detach().cpu().numpy()
                up_imgs = up.detach().cpu().numpy()
                up_preds = output.detach().cpu().numpy()
                for i in range(2):
                    d = down_imgs[i, :3, :, :].transpose(1,2,0)
                    u = up_imgs[i, :, :, :].transpose(1,2,0)
                    pred_u = up_preds[i, :, :, :].transpose(1,2,0)
                    cv2.imwrite(self.result_view_dir + '/test_down_' + str(i) + '.png', d*255)
                    cv2.imwrite(self.result_view_dir + '/test_up_' + str(i) + '.png', u*255)
                    cv2.imwrite(self.result_view_dir + '/test_up_pred_' + str(i) + '.png', pred_u*255)
                evaluate_loss += loss.item()
        evaluate_loss /= batch_num+1
        print('evaluation loss {}'.format(evaluate_loss))


    def parse_data(self, data):
        '''
        Returns a list of the inputs as first output, a list of the GT as a second output, and a list of the remaining info as a third output. Must be implemented.
        '''
        up = data[0].to(self.device)
        down = data[1].to(self.device)
        up_mask = data[3].unsqueeze(1).to(self.device)
        return up, down, up_mask

    def reset_eval_metrics(self):
        '''
        Resets metrics used to evaluate the model
        '''
        self.abs_rel_error_meter.reset()
        self.sq_rel_error_meter.reset()
        self.lin_rms_sq_error_meter.reset()
        self.log_rms_sq_error_meter.reset()
        self.d1_inlier_meter.reset()
        self.d2_inlier_meter.reset()
        self.d3_inlier_meter.reset()
        self.is_best = False


    def compute_eval_metrics(self, output, gt):
        '''
        Computes metrics used to evaluate the model
        '''
        depth_pred = output[0]
        gt_depth = gt[0]
        depth_mask = gt[1]

        N = depth_mask.sum()

        # Align the prediction scales via median
        median_scaling_factor = gt_depth[depth_mask>0].median() / depth_pred[depth_mask>0].median()
        depth_pred *= median_scaling_factor

        abs_rel = abs_rel_error(depth_pred, gt_depth, depth_mask)
        sq_rel = sq_rel_error(depth_pred, gt_depth, depth_mask)
        rms_sq_lin = lin_rms_sq_error(depth_pred, gt_depth, depth_mask)
        rms_sq_log = log_rms_sq_error(depth_pred, gt_depth, depth_mask)
        d1 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=1)
        d2 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=2)
        d3 = delta_inlier_ratio(depth_pred, gt_depth, depth_mask, degree=3)

        self.abs_rel_error_meter.update(abs_rel, N)
        self.sq_rel_error_meter.update(sq_rel, N)
        self.lin_rms_sq_error_meter.update(rms_sq_lin, N)
        self.log_rms_sq_error_meter.update(rms_sq_log, N)
        self.d1_inlier_meter.update(d1, N)
        self.d2_inlier_meter.update(d2, N)
        self.d3_inlier_meter.update(d3, N)


    def load_checkpoint(self, checkpoint_path=None, weights_only=False, eval_mode=False):
        '''
        Initializes network with pretrained parameters
        '''
        if checkpoint_path is not None:
            print('Loading checkpoint \'{}\''.format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)

            # If we want to continue training where we left off, load entire training state
            if not weights_only:
                self.epoch = checkpoint['epoch']
                experiment_name = checkpoint['experiment']
                #self.vis[1] = experiment_name
                self.best_d1_inlier = checkpoint['best_d1_inlier']
                self.loss.from_dict(checkpoint['loss_meter'])
            else:
                print('NOTE: Loading weights only')

            # Load the optimizer and model state
            if not eval_mode:
                util.load_optimizer(
                    self.optimizer, 
                    checkpoint['optimizer'], 
                    self.device)
            util.load_partial_model(
                self.network, 
                checkpoint['state_dict'])
            
            print('Loaded checkpoint \'{}\' (epoch {})'.format(
                checkpoint_path, checkpoint['epoch']))
        else:
            print('WARNING: No checkpoint found')

    def initialize_visualizations(self):
        '''
        Initializes visualizations
        '''

        self.vis[0].line(
            env=self.vis[1],
            X=torch.zeros(1,1).long(),
            Y=torch.zeros(1,1).float(),
            win='losses',
            opts=dict(
                title='Loss Plot',
                markers=False,
                xlabel='Iteration',
                ylabel='Loss',
                legend=['Total Loss']))

        self.vis[0].line(
            env=self.vis[1],
            X=torch.zeros(1,4).long(),
            Y=torch.zeros(1,4).float(),
            win='error_metrics',
            opts=dict(
                title='Depth Error Metrics',
                markers=True,
                xlabel='Epoch',
                ylabel='Error',
                legend=['Abs. Rel. Error', 'Sq. Rel. Error', 'Linear RMS Error', 'Log RMS Error']))

        self.vis[0].line(
            env=self.vis[1],
            X=torch.zeros(1,3).long(),
            Y=torch.zeros(1,3).float(),
            win='inlier_metrics',
            opts=dict(
                title='Depth Inlier Metrics',
                markers=True,
                xlabel='Epoch',
                ylabel='Percent',
                legend=['d1', 'd2', 'd3']))


    def visualize_loss(self, batch_num, loss):
        '''
        Updates the loss visualization
        '''
        total_num_batches = self.epoch * len(self.train_dataloader) + batch_num
        self.vis[0].line(
            env=self.vis[1],
            X=torch.tensor([total_num_batches]),
            Y=torch.tensor([self.loss.avg]),
            win='losses',
            update='append',
            opts=dict(
                legend=['Total Loss']))


    def visualize_samples(self, inputs, gt, other, output):
        '''
        Updates the output samples visualization
        '''
        rgb = inputs[0][0].cpu()
        depth_pred = output[0][0].cpu()
        gt_depth = gt[0][0].cpu()
        depth_mask = gt[1][0].cpu()

        self.vis[0].image(
            visualize_rgb(rgb),
            env=self.vis[1],
            win='rgb',
            opts=dict(
                title='Input RGB Image',
                caption='Input RGB Image'))

        self.vis[0].image(
            visualize_mask(depth_mask),
            env=self.vis[1],
            win='mask',
            opts=dict(
                title='Mask',
                caption='Mask'))


        max_depth = max(
            ((depth_mask>0).float()*gt_depth).max().item(), 
            ((depth_mask>0).float()*depth_pred).max().item())
        self.vis[0].heatmap(
            depth_pred.squeeze().flip(0),
            env=self.vis[1],
            win='depth_pred',
            opts=dict(
                title='Depth Prediction',
                caption='Depth Prediction',
                xmax=max_depth,
                xmin=gt_depth.min().item()))

        self.vis[0].heatmap(
            gt_depth.squeeze().flip(0),
            env=self.vis[1],
            win='gt_depth',
            opts=dict(
                title='Depth GT',
                caption='Depth GT',
                xmax=max_depth))


    def visualize_metrics(self):
        '''
        Updates the metrics visualization
        '''
        abs_rel = self.abs_rel_error_meter.avg
        sq_rel = self.sq_rel_error_meter.avg
        lin_rms = math.sqrt(self.lin_rms_sq_error_meter.avg)
        log_rms = math.sqrt(self.log_rms_sq_error_meter.avg)
        d1 = self.d1_inlier_meter.avg
        d2 = self.d2_inlier_meter.avg
        d3 = self.d3_inlier_meter.avg

        errors = torch.FloatTensor([abs_rel, sq_rel, lin_rms, log_rms])
        errors = errors.view(1, -1)
        epoch_expanded = torch.ones(errors.shape) * (self.epoch+1)
        self.vis[0].line(
            env=self.vis[1],
            X=epoch_expanded,
            Y=errors,
            win='error_metrics',
            update='append',
            opts=dict(
                legend=['Abs. Rel. Error', 'Sq. Rel. Error', 'Linear RMS Error', 'Log RMS Error']))


        inliers = torch.FloatTensor([d1, d2, d3])
        inliers = inliers.view(1, -1)
        epoch_expanded = torch.ones(inliers.shape) * (self.epoch+1)
        self.vis[0].line(
            env=self.vis[1],
            X=epoch_expanded,
            Y=inliers,
            win='inlier_metrics',
            update='append',
            opts=dict(
                legend=['d1', 'd2', 'd3']))

    def print_batch_report(self, batch_num, loss):
        '''
        Prints a report of the current batch
        '''
        print('Epoch: [{0}][{1}/{2}]\t'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
            'Forward Time {forward_time.val:.3f} ({forward_time.avg:.3f})\t'
            'Backward Time {backward_time.val:.3f} ({backward_time.avg:.3f})\n'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\n\n'.format(
            self.epoch+1, 
            batch_num+1,
            len(self.train_dataloader), 
            batch_time=self.batch_time_meter,
            forward_time=self.forward_time_meter, 
            backward_time=self.backward_time_meter, 
            loss=self.loss))

    def print_validation_report(self):
        '''
        Prints a report of the validation results
        '''
        print('Epoch: {}\n'
            '  Avg. Abs. Rel. Error: {:.4f}\n'
            '  Avg. Sq. Rel. Error: {:.4f}\n'
            '  Avg. Lin. RMS Error: {:.4f}\n'
            '  Avg. Log RMS Error: {:.4f}\n'
            '  Inlier D1: {:.4f}\n'
            '  Inlier D2: {:.4f}\n'
            '  Inlier D3: {:.4f}\n\n'.format(
            self.epoch+1, 
            self.abs_rel_error_meter.avg,
            self.sq_rel_error_meter.avg,
            math.sqrt(self.lin_rms_sq_error_meter.avg),
            math.sqrt(self.log_rms_sq_error_meter.avg),
            self.d1_inlier_meter.avg,
            self.d2_inlier_meter.avg,
            self.d3_inlier_meter.avg))

        # Also update the best state tracker
        if self.best_d1_inlier < self.d1_inlier_meter.avg:
            self.best_d1_inlier = self.d1_inlier_meter.avg
            self.is_best = True

    def save_checkpoint(self):
        '''
        Saves the model state
        '''
        # Save latest checkpoint (constantly overwriting itself)
        checkpoint_path = osp.join(
            self.checkpoint_dir, 
            'checkpoint_latest.pth')

        # Actually saves the latest checkpoint and also updating the file holding the best one
        util.save_checkpoint(
            {
                'epoch': self.epoch + 1,
                'experiment': self.name,
                'state_dict': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'loss_meter': self.loss.to_dict(), 
                'best_d1_inlier' : self.best_d1_inlier
            }, 
            self.is_best, 
            filename=checkpoint_path)

        # Copies the latest checkpoint to another file stored for each epoch
        history_path = osp.join(
            self.checkpoint_dir, 
            'checkpoint_{:03d}.pth'.format(self.epoch + 1))
        shutil.copyfile(checkpoint_path, history_path)
        print('Checkpoint saved')


    def save_samples(self, inputs, gt, other, outputs):
        '''
        Saves samples of the network inputs and outputs
        '''
        pass
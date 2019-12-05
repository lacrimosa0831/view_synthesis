
import torch
import torch.utils.data
from util import depth2normal
import numpy as np
from skimage import io
import OpenEXR, Imath, array
import scipy.io
import math
import os.path as osp
from render_view import render_view, depth2disparity

def uv_meshgrid(w, h):
    uv = np.stack(np.meshgrid(range(w), range(h)), axis=-1)
    uv = uv.astype(np.float64)
    uv[..., 0] = ((uv[..., 0] + 0.5) / w - 0.5) * 2 * np.pi
    uv[..., 1] = ((uv[..., 1] + 0.5) / h - 0.5) * np.pi
    return uv

class OmniDepthDataset(torch.utils.data.Dataset):
    '''PyTorch dataset module for effiicient loading'''

    def __init__(self, 
        root_path, 
        path_to_img_list):

        # Set up a reader to load the panos
        self.root_path = root_path

        # Create tuples of inputs/GT
        self.image_list = np.loadtxt(path_to_img_list, dtype=str)

        # Max depth for GT
        self.max_depth = 8.0
    


    def __getitem__(self, idx):
        '''Load the data'''

        # Select the panos to load
        relative_paths = self.image_list[idx]

        # Load the panos
        relative_basename = osp.splitext((relative_paths[0]))[0]
        basename = osp.splitext(osp.basename(relative_paths[0]))[0]
        down = self.readRGBPano(self.root_path + relative_paths[0])
        up = self.readRGBPano(self.root_path + relative_paths[2])
        depth_down = self.readDepthPano(self.root_path + relative_paths[3])
        depth_up = self.readDepthPano(self.root_path + relative_paths[5])
        depth_downmask = ((depth_down <= self.max_depth) & (depth_down > 0.)).astype(np.uint8)
        depth_upmask = ((depth_up <= self.max_depth) & (depth_up > 0.)).astype(np.uint8)
        h, w = depth_up.shape

        uv = uv_meshgrid(w, h)
        uv[...,0] = (uv[...,0]+np.pi)/(2*np.pi)
        uv[...,1] = (uv[...,1]+np.pi/2)/np.pi
        up = np.concatenate([up, uv[...,1:]], 2)
        down = np.concatenate([down, uv[...,1:]], 2)

        # Threshold depths
        depth_up *= depth_upmask
        depth_down *= depth_downmask     
        up = up.astype(np.float32)/255
        down = down.astype(np.float32)/255
        depth_down = np.expand_dims(depth_down, 0)
        # Convert to torch format
        up = torch.from_numpy(up.transpose(2,0,1)).float()  #depth
        down = torch.from_numpy(down.transpose(2,0,1)).float()
        depth_down = torch.from_numpy(depth_down)
        depth_upmask = torch.from_numpy(depth_upmask)
        depth_downmask = torch.from_numpy(depth_downmask)
        
        # Return the set of pano data
        return up, down, depth_down, depth_upmask, depth_downmask
        
    def __len__(self):
        '''Return the size of this dataset'''
        return len(self.image_list)

    def readRGBPano(self, path):
        '''Read RGB and normalize to [0,1].'''
        #rgb = io.imread(path).astype(np.float32) / 255.
        rgb = io.imread(path)
        return rgb


    def readDepthPano(self, path):
        return self.read_exr(path)[...,0].astype(np.float32)
        #mat_content = np.load(path)
        #depth_img = mat_content['depth']
        #return depth_img.astype(np.float32)

    def read_exr(self, image_fpath):
        f = OpenEXR.InputFile( image_fpath )
        dw = f.header()['dataWindow']
        w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)    
        im = np.empty( (h, w, 3) )

        # Read in the EXR
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = f.channels( ["R", "G", "B"], FLOAT )
        for i, channel in enumerate( channels ):
            im[:,:,i] = np.reshape( array.array( 'f', channel ), (h, w) )
        return im
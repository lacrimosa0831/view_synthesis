import math
import numpy as np
import cv2
from skimage import io
from skimage.restoration import inpaint
import scipy.io
import scipy.ndimage
import util

def xyz2uv(xyz):
    normXY = np.sqrt(xyz[:, 0]*xyz[:, 0] + xyz[:, 1]*xyz[:, 1])
    normXY[normXY < 1e-6] = 1e-6
    normXYZ = np.sqrt(xyz[:, 0]*xyz[:, 0] + xyz[:, 1]*xyz[:, 1] + xyz[:, 2]*xyz[:, 2])
    
    v = np.arcsin(xyz[:,2]/normXYZ)
    u = np.arcsin(xyz[:,0]/normXY)
    valid = (xyz[:, 1] < 0) * ( u >= 0)
    u[valid] = math.pi - u[valid]
    valid = (xyz[:, 1] < 0) * ( u <= 0)
    u[valid] = -math.pi - u[valid]
    uv = np.stack([u, v], -1)
    uv[uv!=uv] = 0  #remove nan

    return uv

def uv2coords(uv, w, h):
    w = float(w)
    h = float(h)
    coords = np.zeros_like(uv, dtype = np.float32)
    coords[...,0] = (uv[...,0] + math.pi)/2/math.pi * w + 0.5
    coords[...,1] = (math.pi/2 - uv[...,1])/math.pi * h + 0.5
    coords[...,0] = np.minimum(coords[...,0], w * np.ones_like(coords[...,0], dtype=np.float32))
    coords[...,1] = np.minimum(coords[...,1], h * np.ones_like(coords[...,1], dtype=np.float32))
    coords = np.round(coords).astype(np.int32)
    return coords

def render_view(gt_depth, gt_rgb, perturb=[0, 0, -0.1]):

    depth = np.reshape(gt_depth, [-1, 1])
    depth = np.repeat(depth, 3, 1)
    coords = np.stack(np.meshgrid(range(512), range(256)), -1)
    coords = np.reshape(coords, [-1, 2])
    coords += 1
    uv = util.coords2uv(coords, 512, 256)      
    xyz = util.uv2xyz(uv) 
    new_xyz = xyz * depth

    perturb_xyz = new_xyz.copy()
    # Add a shift to x/y/z axis
    perturb_xyz[:, 0] += perturb[0]
    perturb_xyz[:, 1] += perturb[1]
    perturb_xyz[:, 2] += perturb[2]
    # Get new depth image after shifting
    perturb_new_depth = np.sqrt(perturb_xyz[:, 0]*perturb_xyz[:, 0]
                                        + perturb_xyz[:, 1]*perturb_xyz[:, 1]
                                        + perturb_xyz[:, 2]*perturb_xyz[:, 2])
    perturb_new_depth = np.reshape(perturb_new_depth, [256, 512])
    perturb_uv = xyz2uv(perturb_xyz)
    perturb_coords = uv2coords(perturb_uv, 512, 256)
    # Swap coordinates x and y
    tmp1 = perturb_coords[:, 0]
    tmp2 = perturb_coords[:, 1]
    perturb_coords = np.stack([tmp2, tmp1], -1)
    perturb_coords = np.transpose(perturb_coords)
    perturb_coords -= 1
    # convert coordinates (x, y) to index vector, should be x*num_columns + y
    perturb_idx = perturb_coords[0, :]*512 + perturb_coords[1, :]
    mapped_img = np.zeros([256*512], dtype=np.float32)
    mapped_img[perturb_idx] = perturb_new_depth.flatten()
    mapped_img = np.reshape(mapped_img, [256, 512])

    mapped_r = np.zeros([256*512], dtype=np.uint8)
    mapped_r[perturb_idx] = gt_rgb[:,:,0].flatten()
    mapped_g = np.zeros([256*512], dtype=np.uint8)
    mapped_g[perturb_idx] = gt_rgb[:,:,1].flatten()
    mapped_b = np.zeros([256*512], dtype=np.uint8)
    mapped_b[perturb_idx] = gt_rgb[:,:,2].flatten()
    mapped_rgb = np.stack([mapped_r, mapped_g, mapped_b], -1)
    mapped_rgb = np.reshape(mapped_rgb, [256, 512, 3])

    # apply opencv image inpainting, mask is the area where depth is zero, this is a little time-consuming with large inpaint area.
    mask = mapped_img==0
    #mask = np.expand_dims(mask, 2)
    #mask = np.repeat(mask, 3, axis=2)
    interp_rgb = cv2.inpaint(mapped_rgb, mask.astype(np.uint8), 5, cv2.INPAINT_NS)

    #interp_rgb = inpaint.inpaint_biharmonic(mapped_rgb, mask, multichannel=True)
    """
    interp_img = scipy.ndimage.interpolation.map_coordinates(perturb_new_depth, perturb_coords)
    interp_img = np.reshape(interp_img, [256, 512])
    interp_rgb = np.stack([
           scipy.ndimage.interpolation.map_coordinates(gt_rgb[..., i], perturb_coords, mode='wrap')
            for i in range(gt_rgb.shape[-1])], axis=-1)
    interp_rgb = np.reshape(interp_rgb, [256, 512, 3])
    interp_img = np.flip(interp_img, 0)
    interp_rgb = np.flip(interp_rgb, 0)
    return interp_img, interp_rgb
    """
    return mapped_img, interp_rgb
#scipy.io.savemat('interp.mat',{'interp_img': interp_img, 'gt_depth':gt_depth, 'interp_rgb': interp_rgb})

def depth2disparity(depth, perturb):
    h, w = depth.shape
    angle = np.zeros_like(depth)
    angle2 = np.zeros_like(depth)
    b = perturb[-1]

    for i in range(w):
        for j in range(h):
            theta_T = math.pi - ((j + 0.5) * math.pi/ w)
            angle[j,i] = b* math.sin(theta_T)
            angle2[j,i] = b * math.cos(theta_T)

    disparity = np.zeros_like(depth)
    maskn0 = depth>0
    disparity[maskn0] = angle[maskn0] / (depth[maskn0] - angle2[maskn0])
    disparity[maskn0] = np.arctan(disparity[maskn0])/math.pi * 180
    return disparity

def disparity2depth(disparity, perturb):
    h, w = disparity.shape
    angle = np.zeros_like(disparity)
    angle2 = np.zeros_like(disparity)
    baseline = perturb[-1]

    for i in range(w):
        for j in range(h):
            theta_T = math.pi - ((j+0.5)*math.pi/w)
            angle[j, i] = baseline * math.sin(theta_T)
            angle2[j, i] = baseline * math.cos(theta_T)

    depth = np.zeros_like(disparity)
    mask = disparity>0
    depth[mask] = (angle[mask] / np.tan(disp[mask]/180*math.pi)) + angle2[mask]
    return depth
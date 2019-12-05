import numpy as np
from numpy import sin, cos, tan, pi, arcsin, arctan
from functools import lru_cache
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

# Calculate kernels of SphereCNN
@lru_cache(None)
def get_xy(delta_phi, delta_theta):
    return np.array([
        [
            (-tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1/cos(delta_theta)*tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1),
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1/cos(delta_theta)*tan(delta_phi)),
        ]
    ])

@lru_cache(None)
def cal_index(h, w, img_r, img_c):
    '''
        Calculate Kernel Sampling Pattern
        only support 3x3 filter
        return 9 locations: (3, 3, 2)
    '''
    # pixel -> rad
    phi = -((img_r+0.5)/h*pi - pi/2)
    theta = (img_c+0.5)/w*2*pi-pi

    delta_phi = pi/h
    delta_theta = 2*pi/w

    xys = get_xy(delta_phi, delta_theta)
    x = xys[..., 0]
    y = xys[..., 1]
    rho = np.sqrt(x**2+y**2)
    v = arctan(rho)
    new_phi= arcsin(cos(v)*sin(phi) + y*sin(v)*cos(phi)/rho)
    new_theta = theta + arctan(x*sin(v) / (rho*cos(phi)*cos(v) - y*sin(phi)*sin(v)))
    # rad -> pixel
    new_r = (-new_phi+pi/2)*h/pi - 0.5
    new_c = (new_theta+pi)*w/2/pi - 0.5
    # indexs out of image, equirectangular leftmost and rightmost pixel is adjacent
    new_c = (new_c + w) % w
    new_result = np.stack([new_r, new_c], axis=-1)
    new_result[1, 1] = (img_r, img_c)

    return new_result


@lru_cache(None)
def _gen_filters_coordinates(h, w, stride):
    co = np.array([[cal_index(h, w, i, j) for j in range(0, w, stride)] for i in range(0, h, stride)])
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


def gen_filters_coordinates(h, w, stride=1):
    '''
    return np array of kernel lo (2, H/stride, W/stride, 3, 3)
    '''
    assert(isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates(h, w, stride).copy()


def gen_grid_coordinates(h, w, stride=1):
    coordinates = gen_filters_coordinates(h, w, stride).copy()
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1
    coordinates = coordinates[::-1]
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0]*sz[1], sz[2]*sz[3], sz[4])

    return coordinates.copy()


class SphereConv2D(nn.Module):
    '''  SphereConv2D
    Note that this layer only support 3x3 filter
    '''
    def __init__(self, in_c, out_c, stride=1, bias=True, mode='bilinear'):
        super(SphereConv2D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode
        self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        self.bias.data.zero_()

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        x = nn.functional.grid_sample(x, grid, mode=self.mode)
        x = nn.functional.conv2d(x, self.weight, self.bias, stride=3)
        return x


class SphereMaxPool2D(nn.Module):
    '''  SphereMaxPool2D
    Note that this layer only support 3x3 filter
    '''
    def __init__(self, stride=1, mode='bilinear'):
        super(SphereMaxPool2D, self).__init__()
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[2:4]):
            self.grid_shape = tuple(x.shape[2:4])
            coordinates = gen_grid_coordinates(x.shape[2], x.shape[3], self.stride)
            with torch.no_grad():
                self.grid = torch.FloatTensor(coordinates).to(x.device)
                self.grid.requires_grad = True

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1)

        return self.pool(nn.functional.grid_sample(x, grid, mode=self.mode))

class SphericalUnet(nn.Module):
    def __init__(self):
        super(SphericalUnet, self).__init__()
        self.conv0_0 = SphereConv2D(3, 32, stride=1)
        self.conv0_1 = SphereConv2D(32, 64)
        self.pool0 = SphereMaxPool2D(stride=2)
        self.conv1_0 = SphereConv2D(64, 128)
        self.conv1_1 = SphereConv2D(128, 128)
        self.pool1 = SphereMaxPool2D(stride=2)
        self.conv2_0 = SphereConv2D(128, 256)
        self.conv2_1 = SphereConv2D(256, 256)
        self.conv2_2 = SphereConv2D(256, 256)
        self.conv2_3 = SphereConv2D(256, 512)
        self.conv2_4 = SphereConv2D(512, 512)

        self.decoder0 = SphereConv2D(512+128, 256, stride=1)
        self.decoder1 = SphereConv2D(256, 256, stride=1)
        self.decoder2 = SphereConv2D(256+64, 128, stride=1)
        self.decoder3 = SphereConv2D(128, 128, stride=1)
        self.decoder4 = SphereConv2D(128, 64, stride=1)
        self.prediction = nn.Conv2d(64, 3, 1)
        self.activate = nn.ELU()

    def forward(self, x):
        conv0_0 = self.activate(self.conv0_0(x))
        conv0_1 = self.activate(self.conv0_1(conv0_0))
        pool0 = self.pool0(conv0_1)
        conv1_0 = self.activate(self.conv1_0(pool0))
        conv1_1 = self.activate(self.conv1_1(conv1_0))
        pool1 = self.pool1(conv1_1)
        conv2_0 = self.activate(self.conv2_0(pool1))
        conv2_1 = self.activate(self.conv2_1(conv2_0))
        conv2_2 = self.activate(self.conv2_2(conv2_1))
        conv2_3 = self.activate(self.conv2_3(conv2_2))
        conv2_4 = self.activate(self.conv2_4(conv2_3))

        upsample0 = F.interpolate(conv2_4, scale_factor=2)
        decoder0 = self.activate(self.decoder0(torch.cat([upsample0, conv1_1], 1)))
        decoder1 = self.activate(self.decoder1(decoder0))
        upsample1 = F.interpolate(decoder0, scale_factor=2)
        decoder2 = self.activate(self.decoder2(torch.cat([upsample1, conv0_1], 1)))
        decoder3 = self.activate(self.decoder3(decoder2))
        decoder4 = self.activate(self.decoder4(decoder3))

        prediction = self.prediction(decoder4)
        return prediction


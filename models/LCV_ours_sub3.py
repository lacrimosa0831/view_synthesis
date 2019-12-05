from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
#from sub_ASPP import *

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         #nn.BatchNorm2d(out_planes))
                         nn.GroupNorm(out_planes//4, out_planes))
    

def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         #nn.BatchNorm3d(out_planes))
                         nn.GroupNorm(out_planes//4, out_planes))
    

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left  = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out

class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        # ASPP network

        self.aspp1 = nn.Sequential(convbn(160, 32, 1, 1, 0, 1),
                    nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(convbn(160, 32, 3, 1, 1, 6),
                    nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(convbn(160, 32, 3, 1, 1, 12),
                    nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(convbn(160, 32, 3, 1, 1, 18),
                    nn.ReLU(inplace=True))
        self.aspp5 = nn.Sequential(convbn(160, 32, 3, 1, 1, 24),
                    nn.ReLU(inplace=True))
        self.newlastconv = nn.Sequential(convbn(224, 128, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))
        # Polar Branch
        self.firstcoord = nn.Sequential(convbn(1, 32, 3, 2, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 2, 1, 1),
                                        nn.ReLU(inplace=True),
                                        convbn(32, 32, 3, 1, 1, 1),
                                        nn.ReLU(inplace=True))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(planes * block.expansion),)
                nn.GroupNorm(planes * block.expansion//4, planes * block.expansion),)
            
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x[:,:3,:,:])
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        # coord avg pooling and concat to main feature
        out_coord = self.firstcoord(x[:,3:,:,:])
        output_skip_c = torch.cat((output_skip, out_coord), 1)

        # ASPP and new last conv
        ASPP1 = self.aspp1(output_skip_c)
        ASPP2 = self.aspp2(output_skip_c)
        ASPP3 = self.aspp3(output_skip_c)
        ASPP4 = self.aspp4(output_skip_c)
        ASPP5 = self.aspp5(output_skip_c)
        output_feature = torch.cat((output_raw, ASPP1,ASPP2,ASPP3,ASPP4,ASPP5), 1)
        output_feature = self.newlastconv(output_feature)
	
        return output_feature

class forfilter(nn.Module):
    def __init__(self, inplanes):
        super(forfilter, self).__init__()
    
        self.forfilter1 = nn.Conv2d(1, 1, (7, 1), 1, (0, 0), bias=False )
        self.inplanes = inplanes
    
    def forward(self, x):
    
        out = self.forfilter1( F.pad( torch.unsqueeze( x[:,0,:,:], 1), pad=(0,0,3,3), mode='replicate'))
        for i in range(1, self.inplanes):
            out = torch.cat( (out, self.forfilter1( F.pad( torch.unsqueeze( x[:,i,:,:], 1), pad=(0,0,3,3), mode='replicate'))), 1)
    
        return out

class disparityregression_sub3(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression_sub3, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp *3 )),[1,maxdisp *3,1,1]) /3 ).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=(1, 1, 1), stride=2,bias=False),
                                  # nn.BatchNorm3d(inplanes*2)) #+conv2, single value on side length image 
                                   nn.GroupNorm(inplanes//2, inplanes*2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=(0, 1, 1), stride=2,bias=False),
                                   #nn.BatchNorm3d(inplanes)) #+x
                                   nn.GroupNorm(inplanes//4, inplanes))

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class LCV(nn.Module):
    def __init__(self, maxdisp):
        super(LCV, self).__init__()
        self.maxdisp = maxdisp

        self.feature_extraction = feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        # for loop of filter kernel
        self.forF = forfilter(32)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, up, down):

        refimg_fea     = self.feature_extraction(up)              # reference image feature
        targetimg_fea  = self.feature_extraction(down)             # target image feature

        #matching
        cost = Variable(torch.FloatTensor(refimg_fea.shape[0], refimg_fea.shape[1]*2, self.maxdisp//4 *3, 
             refimg_fea.shape[2],  refimg_fea.shape[3]).zero_()).cuda()

        for i in range(self.maxdisp//4 *3):
            if i > 0 :
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea[:,:,:,:]
                cost[:, refimg_fea.size()[1]:, i, :,:] = shift_down[:,:,:,:]
                shift_down = self.forF(shift_down)
            else:
                cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
                shift_down = self.forF(targetimg_fea)

        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        cost1 = F.upsample(cost1, [self.maxdisp *3,up.size()[2],up.size()[3]], mode='trilinear')    # when within units, the maxdisp needs to be modified
        cost2 = F.upsample(cost2, [self.maxdisp *3,up.size()[2],up.size()[3]], mode='trilinear')

        cost1 = torch.squeeze(cost1,1)
        pred1 = F.softmax(cost1,dim=1)
        pred1 = disparityregression_sub3(self.maxdisp)(pred1)

        cost2 = torch.squeeze(cost2,1)
        pred2 = F.softmax(cost2,dim=1)
        pred2 = disparityregression_sub3(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp *3,up.size()[2],up.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        pred3 = disparityregression_sub3(self.maxdisp)(pred3)

        return pred1, pred2, pred3

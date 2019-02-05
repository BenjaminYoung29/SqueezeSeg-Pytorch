""" SqueezeSeg Model """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bilateral_filter import BilateralFilter
from .recurrent_crf import RecurrentCRF

class Conv( nn.Module ):
    def __init__( self, inputs, outputs, kernel_size=3, stride=1, padding=0 ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        return F.relu(self.conv(x))


class MaxPool( nn.Module ):
    def __init__( self, kernel_size=3, stride=1, padding=0 ):
        super(MaxPool, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=True)

    def forward(self,x):
        return self.pool(x)


class Fire( nn.Module ):
    def __init__( self, inputs, o_sq1x1, o_ex1x1, o_ex3x3 ):
        """ Fire layer constructor.

        Args:
            inputs : input tensor
            o_sq1x1 : output of squeeze layer
            o_ex1x1 : output of expand layer(1x1)
            o_ex3x3 : output of expand layer(3x3)
        """
        super(Fire, self).__init__()
        self.sq1x1 = Conv(inputs, o_sq1x1, kernel_size=1, stride=1, padding=0)
        self.ex1x1 = Conv(o_sq1x1, o_ex1x1, kernel_size=1, stride=1, padding=0)
        self.ex3x3 = Conv(o_sq1x1, o_ex3x3, kernel_size=3, stride=1, padding=1)
    
    def forward(self,x):
        return torch.cat([self.ex1x1( self.sq1x1(x) ), self.ex3x3( self.sq1x1(x) )], 1)


class Deconv( nn.Module ):
    def __init__( self, inputs, outputs, kernel_size, stride, padding=0 ):
        super(Deconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return F.relu(self.deconv(x))


class FireDeconv( nn.Module ):
    def __init__( self, inputs, o_sq1x1, o_ex1x1, o_ex3x3 ):
        super(FireDeconv, self).__init__()
        self.sq1x1 = Conv(inputs, o_sq1x1, 1, 1, 0)
        self.deconv = Deconv(o_sq1x1, o_sq1x1, [1,4], [1,2], [0,1])
        self.ex1x1 = Conv(o_sq1x1, o_ex1x1, 1, 1, 0)
        self.ex3x3 = Conv(o_sq1x1, o_ex3x3, 3, 1, 1)

    def forward(self,x):
        x = self.sq1x1(x)
        x = self.deconv(x)
        return torch.cat( [self.ex1x1(x), self.ex3x3(x)], 1)

class SqueezeSeg( nn.Module ):
    # __init__(引数)　後で考える drop率とかかな
    def __init__(self, mc):
        super(SqueezeSeg, self).__init__()
        
        # config
        self.mc = mc

        # encoder
        self.conv1 = Conv(5, 64, 3, (1,2), 1)
        self.conv1_skip = Conv(5, 64, 1, 1, 0)
        self.pool1 = MaxPool(3, (1,2), (1,0))
        
        self.fire2 = Fire(64, 16, 64, 64)
        self.fire3 = Fire(128, 16, 64, 64)
        self.pool3 = MaxPool(3, (1,2), (1,0))
        
        self.fire4 = Fire(128, 32, 128, 128)
        self.fire5 = Fire(256,32,128,128)
        self.pool5 = MaxPool(3, (1,2), (1,0))
        
        self.fire6 = Fire(256, 48, 192, 192)
        self.fire7 = Fire(384, 48, 192, 192)
        self.fire8 = Fire(384, 64, 256, 256)
        self.fire9 = Fire(512, 64, 256, 256)

        # decoder
        self.fire10 = FireDeconv(512, 64, 128, 128)
        self.fire11 = FireDeconv(256, 32, 64, 64)
        self.fire12 = FireDeconv(128, 16, 32, 32)
        self.fire13 = FireDeconv(64, 16, 32, 32)
        
        self.drop = nn.Dropout2d()

        # reluを適用させない
        self.conv14 = nn.Conv2d(64, mc.NUM_CLASS, kernel_size=3, stride=1, padding=1)

        self.bf = BilateralFilter(mc, stride=1, padding=(1,2))

        self.rc = RecurrentCRF(mc, stride=1, padding=(1,2))

    def forward(self, x, lidar_mask):
        # encoder
        out_c1 = self.conv1(x)
        out = self.pool1(out_c1)
        
        out_f3 = self.fire3( self.fire2(out) )
        out = self.pool3(out_f3)

        out_f5 = self.fire5( self.fire4(out) )
        out = self.pool5(out_f5)

        out = self.fire9( self.fire8( self.fire7( self.fire6(out) ) ) )
        
        # decoder
        out = torch.add(self.fire10(out), out_f5)
        out = torch.add(self.fire11(out), out_f3)
        out = torch.add(self.fire12(out), out_c1)
        out = self.drop( torch.add(self.fire13(out), self.conv1_skip(x)) )
        out = self.conv14(out)
        
        bf_w = self.bf(x[:, :3, :, :])

        out = self.rc(out, lidar_mask, bf_w)

        return out



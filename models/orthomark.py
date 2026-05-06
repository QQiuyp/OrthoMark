import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return [out.narrow(1, 0, 3), out.narrow(1, 3, 9)]

        else:
            x_cat = torch.cat((x[0], x[1]), 1)
            out = x_cat.reshape([x_cat.shape[0], 4, self.channel_in, x_cat.shape[2], x_cat.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x_cat.shape[0], self.channel_in * 4, x_cat.shape[2], x_cat.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)


class DB(nn.Module):
    def __init__(self, input, output, mid_dim=32, bias=True):
        super(DB, self).__init__()
        self.mid_dim = mid_dim
        self.conv1 = nn.Conv2d(input, self.mid_dim, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + self.mid_dim, self.mid_dim, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * self.mid_dim, self.mid_dim, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * self.mid_dim, self.mid_dim, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * self.mid_dim, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class RFE_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__()

        self.clamp = clamp
        self.r = DB(input=3, output=9)
        self.y = DB(input=3, output=9)
        self.f = DB(input=9, output=3)

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):

        x1, x2 = x[0], x[1]

        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2

            s1, t1 = self.r(y1), self.y(y1)

            y2 = self.e(s1) * x2 + t1

            out = [y1, y2]

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)

            t2 = self.f(y2)
            y1 = x1 - t2

            out = [y1, y2]
        return out


class RFE(nn.Module):
    def __init__(self):
        super(RFE, self).__init__()

        self.inv1 = RFE_block()
        self.inv2 = RFE_block()
        self.inv3 = RFE_block()
        self.inv4 = RFE_block()

    def forward(self, x, rev=False):
        if rev:
            out = self.inv4(x, rev)
            out = self.inv3(out, rev)
            out = self.inv2(out, rev)
            out = self.inv1(out, rev)
        else:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
        return out


class OrthoMark(nn.Module):
    def __init__(self):
        super(OrthoMark, self).__init__()
        self.model = RFE()
        self.haar = HaarDownsampling(3)

    def forward(self, x, rev=False):
        out = self.haar(x)
        out = self.model(out, rev)
        out = self.haar(out, rev=True)
        return out

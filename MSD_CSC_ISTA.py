import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MSD_CSC_net(nn.Module):
    def __init__(self,k,d,in_channels=None,class_num=None):
        super(MSD_CSC_net, self).__init__()

        self.k = k
        self.d = d
        self.in_channels = in_channels
        self.ini_channels = 16
        self.class_num = class_num

        self.filter0 = nn.Parameter(torch.randn(self.ini_channels, self.in_channels, 3, 3), requires_grad=True)
        self.filter1 = nn.Parameter(torch.randn(160, 160, 1, 1), requires_grad=True)
        self.filter2 = nn.Parameter(torch.randn(304, 304, 1, 1), requires_grad=True)

        self.filters1 = nn.ParameterList(
            [nn.Parameter(torch.randn(self.k, self.ini_channels + self.k * i, 3, 3), requires_grad=True) for i in
             range(self.d)])
        self.filters2 = nn.ParameterList(
            [nn.Parameter(torch.randn(self.k, 160 + self.k * i, 3, 3), requires_grad=True) for i in
             range(self.d)])
        self.filters3 = nn.ParameterList(
            [nn.Parameter(torch.randn(self.k, 304 + self.k * i, 3, 3), requires_grad=True) for i in
             range(self.d)])

        self.b0 = nn.Parameter(torch.zeros(1, self.ini_channels, 1, 1), requires_grad=True)

        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.ini_channels + self.k + self.k * i, 1, 1), requires_grad=True) for i in
             range(self.d)])
        self.b2 = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 160 + self.k + self.k * i, 1, 1), requires_grad=True) for i in
             range(self.d)])
        self.b3 = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 304 + self.k + self.k * i, 1, 1), requires_grad=True) for i in
             range(self.d)])

        self.bn0 = nn.BatchNorm2d(self.ini_channels, affine=True).cuda()
        self.bn1 = [nn.BatchNorm2d(self.ini_channels+(i+1)*self.k, affine=True).cuda() for i in
                    range(self.d)]
        self.bn2 = [nn.BatchNorm2d(160 + (i + 1)* self.k, affine=True).cuda() for i in
                    range(self.d)]
        self.bn3 = [nn.BatchNorm2d(304 + (i + 1)* self.k, affine=True).cuda() for i in
                    range(self.d)]

        self.c1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True) for i in range(self.d)])
        self.c2 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True) for i in range(self.d)])
        self.c3 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, 1, 1, 1), requires_grad=True) for i in range(self.d)])

        # classifier
        self.Wclass = nn.Linear(448,self.class_num)

        # Initialization
        for i in range(self.d):
            self.filters1[i].data = .1 / np.sqrt((self.ini_channels + self.k * i) * 9) * self.filters1[i].data
            self.filters2[i].data = .1 / np.sqrt((160 + self.k * i) * 9) * self.filters2[i].data
            self.filters3[i].data = .1 / np.sqrt((304 + self.k * i) * 9) * self.filters3[i].data
        self.filter0.data = .1 / np.sqrt(self.in_channels * 9) * self.filter0.data
        self.filter1.data = .1 / np.sqrt(160) * self.filter1.data
        self.filter2.data = .1 / np.sqrt(304) * self.filter2.data

    def MSD_CSC_ISTA_Block(self, input, k, d, filters, b, bn, c, dilation_cycle, unfolding):
        features = []
        features.append(input)

        for i in range(d):
            f1 = F.conv2d(features[-1], filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                          dilation=(i % dilation_cycle) + 1)
            f2 = torch.cat((features[-1], f1), dim=1)
            del f1
            f3 = c[i] * f2 + b[i]
            del f2
            features.append(F.relu(bn[i](f3)))
            del f3

        # backward
        for loop in range(unfolding):
            for i in range(d - 1):
                f1 = F.conv_transpose2d(features[-1 - i][:, -k:, :, :], filters[-1 - i], stride=1,
                                        padding=((-1 - i + d) % dilation_cycle) + 1,
                                        dilation=((-1 - i + d) % dilation_cycle) + 1)
                features[-2 - i] = f1 + features[-1 - i][:, 0:-k, :, :]
            # forward
            #print("forward")
            for i in range(d):
                #print(i)
                f1 = F.conv_transpose2d(features[i + 1][:, -k:, :, :], filters[i], stride=1,
                                        padding=(i % dilation_cycle) + 1, dilation=(i % dilation_cycle) + 1)
                f2 = features[i + 1][:, 0:-k, :, :] + f1
                del f1
                f3 = F.conv2d(f2, filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f4 = torch.cat((f2, f3), dim=1)  ###
                del f2,f3
                f5 = F.conv2d(features[i], filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f6 = torch.cat((features[i], f5), dim=1)  ###
                f7 = features[i + 1] - c[i] * (f4 - f6) + b[i]
                del f4,f6
                features[i + 1] = F.relu(bn[i](f7))

        return features[-1]

    def MSD_CSC_FISTA_Block(input, k, d, filters, b, bn, c, dilation_cycle, unfolding):
        t = 1
        t_prv = t
        # Encoding
        features = []
        features.append(input)
        for i in range(d):
            f1 = F.conv2d(features[-1], filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                          dilation=(i % dilation_cycle) + 1)
            f2 = torch.cat((features[-1], f1), dim=1)
            del f1
            f3 = c[i] * f2 + b[i]
            del f2
            features.append(F.relu(bn[i](f3)))
            del f3
        feature_prv = features[-1]

        for loop in range(unfolding):

            t_prv = t
            t = float((1 + np.sqrt(1 + 4 * t_prv ** 2)) / 2)

            Z = features[-1] + (t_prv - 1) / t * (features[-1] - feature_prv)
            feature_prv = features[-1]
            features[-1] = Z

            # backward
            for i in range(d - 1):
                f1 = F.conv_transpose2d(features[-1 - i][:, -k:, :, :], filters[-1 - i], stride=1,
                                        padding=((-1 - i + d) % dilation_cycle) + 1,
                                        dilation=((-1 - i + d) % dilation_cycle) + 1)
                features[-2 - i] = f1 + features[-1 - i][:, 0:-k, :, :]

            # forward
            for i in range(d):
                #print(i)
                f1 = F.conv_transpose2d(features[i + 1][:, -k:, :, :], filters[i], stride=1,
                                        padding=(i % dilation_cycle) + 1, dilation=(i % dilation_cycle) + 1)
                f2 = features[i + 1][:, 0:-k, :, :] + f1
                del f1
                f3 = F.conv2d(f2, filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f4 = torch.cat((f2, f3), dim=1)  ###
                del f2, f3
                f5 = F.conv2d(features[i], filters[i], stride=1, padding=(i % dilation_cycle) + 1,
                              dilation=(i % dilation_cycle) + 1)
                f6 = torch.cat((features[i], f5), dim=1)  ###
                f7 = features[i + 1] - c[i] * (f4 - f6) + b[i]
                del f4, f6
                features[i + 1] = F.relu(bn[i](f7))

        return features[-1]

    def forward(self, x):
        x = F.conv2d(x, self.filter0, stride=1,padding=1) + self.b0
        x = self.bn0(x)
        x = F.relu(x)

        x = self.MSD_CSC_ISTA_Block(x,12,12,self.filters1,self.b1,self.bn1,self.c1,6,1)

        x = F.conv2d(x, self.filter1, stride=1,padding=0)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.MSD_CSC_ISTA_Block(x, 12, 12, self.filters2, self.b2, self.bn2, self.c2, 6, 1)

        x = F.conv2d(x, self.filter2, stride=1, padding=0)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)

        x = self.MSD_CSC_ISTA_Block(x, 12, 12, self.filters3, self.b3, self.bn3, self.c3, 6, 1)

        x = F.avg_pool2d(x, kernel_size = 8, stride=1, padding=0)

        x = x.view(x.shape[0], -1)
        x = self.Wclass(x)
        output = F.log_softmax(x, dim=1)

        return output

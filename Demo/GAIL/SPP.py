#//////////////#####///////////////
#
# ANU u6325688 Yangyang Xu
# Supervisor: Dr.Penny Kyburz
# SPP used in this scrip is adopted some methods from :
# https://github.com/yueruchen/sppnet-pytorch/blob/master/cnn_with_spp.py
#//////////////#####///////////////
import math
import torch
import torch.nn as nn
class SPP(nn.Module):

    def __init__(self):
        super(SPP, self).__init__()

    def doSPP(self, prevConv, numSample, prevConvSize, outPoolSize, kernelSize):

        for i in range(len(outPoolSize)):
            # print(outPoolSize)
            w_ker = int(math.ceil(prevConvSize[0] / outPoolSize[i]))
            h_ker = int(math.ceil(prevConvSize[1] / outPoolSize[i]))
            w_str = int(math.floor(prevConvSize[0] / outPoolSize[i]))
            h_str = int(math.floor(prevConvSize[1] / outPoolSize[i]))
            w_pad = int(math.floor((w_ker-1) / 2))
            h_pad = int(math.floor((h_ker-1) / 2))
            #w_pad = (h_wid * outPoolSize[i] - prevConvSize[0] + 1) / 2
            #h_pad = (w_wid * outPoolSize[i] - prevConvSize[1] + 1) / 2
            #w_pad = (h_wid * outPoolSize[i] - prevConvSize[0] + 1) / 2
            #h_pad = (w_wid * outPoolSize[i] - prevConvSize[1] + 1) / 2

            # pad should be smaller than 1/2 kernel size
            maxpool = nn.MaxPool2d((h_ker,w_ker), stride=(h_str, w_str), padding=(h_pad, w_pad))

            x = maxpool(prevConv)
            if (i == 0):
                spp = x.view(numSample, -1)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp, x.view(numSample, -1)), 1)
        return spp




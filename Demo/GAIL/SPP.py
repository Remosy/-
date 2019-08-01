import math
import torch
import torch.nn as nn
class SPP:

    def __init__(self, prevConv, numSample, prevConvSize, outPoolSize):
        super(SPP, self).__init__()
        for i in range(len(outPoolSize)):
            # print(outPoolSize)
            w_wid = int(math.ceil(prevConvSize[0] / outPoolSize[i]))
            h_wid = int(math.ceil(prevConvSize[1] / outPoolSize[i]))
            w_pad = (h_wid * outPoolSize[i] - prevConvSize[0] + 1) / 2
            h_pad = (w_wid * outPoolSize[i] - prevConvSize[1] + 1) / 2
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            x = maxpool(prevConv)
            if (i == 0):
                spp = x.view(numSample, -1)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp, x.view(numSample, -1)), 1)
        return spp


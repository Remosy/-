"""
Policy Critic
"""

import torch.nn as nn
from GAIL.SPP import SPP
from commons.DataInfo import DataInfo
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Critic(nn.Module):
    """The summary line for a class docstring should fit on one line.

        If the class has public attributes, they may be documented here
        in an ``Attributes`` section and follow the same formatting as a
        function's ``Args`` section. Alternatively, attributes may be documented
        inline with the attribute's declaration (see __init__ method below).

        Properties created with the ``@property`` decorator should be documented
        in the property's getter method.

        Attributes:
            attr1 (str): Description of `attr1`.
            attr2 (:obj:`int`, optional): Description of `attr2`.

        """
    def __init__(self, datainfo:DataInfo):
        super(Critic, self).__init__()
        self.inChannel = datainfo.generatorIn #state space size
        self.outChannel = 1 #a value
        self.kernel = datainfo.generatorKernel #number of filter
        self.maxAction = datainfo.maxAction
        self.pyramidLevel = [4, 2, 1]  # 3-level pyramid
        self.spp = SPP().to(device)

        self.main = nn.Sequential(
            nn.Conv2d(self.inChannel, self.outChannel, kernel_size=self.kernel, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.outChannel * 2),
            nn.ReLU(True),
            nn.Linear(self.outChannel * 8, self.outChannel)
        )

        self.fc2 =

    def forward(self, input):
        output = self.main(input)
        return output


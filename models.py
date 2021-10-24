import torch.nn as nn
import torch
import numpy as np


import torch.nn as nn
import torch
import numpy as np

class ChessModelBase(nn.Module):
    def __init__(self, in_channels:int=41):
        super().__init__()

        self.first = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1, padding=0)
        # self.bnorm = nn.BatchNorm2d(256) 
        self.s1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=3)
        self.classify = nn.Linear(1024, 1965)
        self.relu = nn.ReLU()

    def forward(self, x):
        xprime = self.first(x)
        x1 = self.relu(xprime)
        # x1 = self.bnorm(x1)
        x2 = self.relu(self.s1(x1))
        x3 = self.relu(self.s2(x2))
        x3 = self.relu(self.s3(x3)) + x2
        x4 = self.relu(self.s4(x3))
        x4 = self.relu(self.s5(x4)) + x3
        x5 = self.relu(self.s6(x4))
        x6 = self.relu(self.s7(x5))
        x7 = self.pool(x6)
        logits = self.classify(torch.reshape(x7, (-1,1024)))
        out = nn.functional.softmax(logits, dim=1)
        return out



class ChessModel(nn.Module):
    def __init__(self, in_channels:int=41):
        super().__init__()

        self.first = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1, padding=0)
        # self.bnorm = nn.BatchNorm2d(256) 
        self.s1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.s8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.classify1 = nn.Linear(9216, 1965)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        xprime = self.first(x)
        x1 = self.relu(xprime)
        # x1 = self.bnorm(x1)
        x2 = self.relu(self.s1(x1))
        x3 = self.relu(self.s2(x2))
        x3 = self.relu(self.s3(x3)) + x2
        x4 = self.relu(self.s4(x3))
        x4 = self.relu(self.s5(x4)) + x3
        x5 = self.relu(self.s6(x4))
        x6 = self.relu(self.s7(x5)) + x4
        x7 = self.relu(self.s8(x6))
        logits = self.relu(self.classify1(torch.reshape(x7, (-1,9216))))
        out = nn.functional.softmax(logits, dim=1)
        return out
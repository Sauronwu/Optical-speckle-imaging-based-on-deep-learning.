import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class net_model(nn.Module):
    def __init__(self):
        super(net_model, self).__init__()

        self.layer_01 = torch.nn.Sequential(
            torch.nn.Conv2d (
                1, 32, (3, 3), stride=(1, 1), padding=1,# 输出大小为32*256*256
            ),
            torch.nn.BatchNorm2d(32,affine=True),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1),#输出大小为32*256*256
            torch.nn.BatchNorm2d(32,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_01_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)  # 输出大小为32*128*128
        )

        self.layer_02 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(64,affine=True),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1),#输出大小为64*128*128
            torch.nn.BatchNorm2d(64,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_02_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )#输出大小为64*64*64

        self.layer_03 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(128,affine=True),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(128,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_03_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )

        self.layer_04 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(256,affine=True),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(256,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_04_maxpool = torch.nn.MaxPool2d(
            kernel_size=(2, 2), stride=(2, 2)
        )

        self.layer_05 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(512,affine=True),
            torch.nn.LeakyReLU(),

            torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(512,affine=True),
            torch.nn.LeakyReLU(),
        )

        # self.layer_part1 = torch.nn.Sequential(
        #     self.layer_01, self.layer_01_maxpool,
        #     self.layer_02, self.layer_02_maxpool,
        #     self.layer_03, self.layer_03_maxpool,
        #     self.layer_04, self.layer_04_maxpool, self.layer_05
        # )

        # -------------------------------------------------------

        # layer_06

        self.layer_06_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                512, 256, (3, 3), stride=(2, 2), padding=1,
                output_padding=1 # 输出大小为256*128*128
            ),#
            torch.nn.BatchNorm2d(256,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_06_02 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, (3, 3), stride=(1, 1), padding=1),#输出大小为256*128*128
            torch.nn.BatchNorm2d(256,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_06_03 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1), #输出大小为256*128*128
            torch.nn.BatchNorm2d(256,affine=True),
            torch.nn.LeakyReLU(),
        )

        # layer_07

        self.layer_07_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                256, 128, (3, 3), stride=(2, 2), padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(128,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_07_02 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(128,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_07_03 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(128,affine=True),
            torch.nn.LeakyReLU(),
        )

        # layer_08

        self.layer_08_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                128, 64, (3, 3), stride=(2, 2), padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(64,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_08_02 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=1),
            torch.nn.BatchNorm2d(64,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_08_03 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1,),
            torch.nn.BatchNorm2d(64,affine=True),
            torch.nn.LeakyReLU(),
        )

        # layer_09

        self.layer_09_01 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                64, 32, (3, 3), stride=(2, 2), padding=1,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(32,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_09_02 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=1,),
            torch.nn.BatchNorm2d(32,affine=True),
            torch.nn.LeakyReLU(),
        )

        self.layer_09_03 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1,),
            torch.nn.BatchNorm2d(32,affine=True),
            torch.nn.LeakyReLU(),
        )

        # layer_10

        self.layer_10 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, (3, 3), stride=(1, 1), padding=1,),
            torch.nn.BatchNorm2d(1,affine=True),
            torch.nn.ReLU()  #
        )

    def forward(self, x):
        # -------------------------------------------------------
        L1 = self.layer_01(x)
        L2 = self.layer_02(self.layer_01_maxpool(L1))
        L3 = self.layer_03(self.layer_02_maxpool(L2))
        L4 = self.layer_04(self.layer_03_maxpool(L3))
        L5 = self.layer_05(self.layer_04_maxpool(L4))
        L6_1 = self.layer_06_01(L5)
        L6_M = torch.cat((L6_1, L4), 1)
        L6_2 = self.layer_06_02(L6_M)
        L6 = self.layer_06_03(L6_2)
        L7_1 = self.layer_07_01(L6)
        L7_M = torch.cat((L7_1, L3), 1)
        L7_2 = self.layer_07_02(L7_M)
        L7 = self.layer_07_03(L7_2)
        L8_1 = self.layer_08_01(L7)
        L8_M = torch.cat((L8_1, L2), 1)
        L8_2 = self.layer_08_02(L8_M)
        L8 = self.layer_08_03(L8_2)
        L9_1 = self.layer_09_01(L8)
        L9_M = torch.cat((L9_1, L1), 1)
        L9_2 = self.layer_09_02(L9_M)
        L9 = self.layer_09_03(L9_2)
        L10 = self.layer_10(L9)

        return L10
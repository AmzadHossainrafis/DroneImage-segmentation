import torch
import torch.nn as nn


class Unet(nn.Module):
    """
    this is the unet model implementation base on paper https://arxiv.org/pdf/1505.04597.pdf

    args:
        n_class: number of classes --> in our case 22
        in_channels: number of input channels --> in our case 3
        filters: list of filters for each layer --> in our case [64, 128, 256, 512, 1024] --> you can change this according to
        your need but make sure the number of filters are in the power of 2 and change encoder and decoder accordingly.

    return:
        x42: output of the model

    """

    def __init__(self, n_class=22, in_channels=3, filters=[64, 128, 256, 512, 1024]):
        super(Unet, self).__init__()

        # Encoder

        self.conv1 = nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        # reshape to [1, 512, 70, 70]
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(filters[4], filters[4], kernel_size=3, padding=1)
        self.relu10 = nn.ReLU(inplace=True)

        # Decoder

        self.trans1 = nn.ConvTranspose2d(
            filters[4], filters[3], kernel_size=2, stride=2
        )
        self.conv11 = nn.Conv2d(filters[4], filters[3], kernel_size=3, padding=1)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1)
        self.relu12 = nn.ReLU(inplace=True)

        self.trans2 = nn.ConvTranspose2d(
            filters[3], filters[2], kernel_size=2, stride=2
        )
        self.conv13 = nn.Conv2d(filters[3], filters[2], kernel_size=3, padding=1)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv14 = nn.Conv2d(filters[2], filters[2], kernel_size=3, padding=1)

        self.trans3 = nn.ConvTranspose2d(
            filters[2], filters[1], kernel_size=2, stride=2
        )
        self.conv15 = nn.Conv2d(filters[2], filters[1], kernel_size=3, padding=1)
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1)

        self.trans4 = nn.ConvTranspose2d(
            filters[1], filters[0], kernel_size=2, stride=2
        )
        self.conv17 = nn.Conv2d(filters[1], filters[0], kernel_size=3, padding=1)
        self.relu17 = nn.ReLU(inplace=True)
        self.conv18 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)

        self.conv19 = nn.Conv2d(filters[0], n_class, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.conv2(x2)
        x4 = self.relu2(x3)
        x5 = self.maxpool(x4)

        x6 = self.conv3(x5)
        x7 = self.relu3(x6)
        x8 = self.conv4(x7)
        x9 = self.relu4(x8)
        x10 = self.maxpool2(x9)

        x11 = self.conv5(x10)
        x12 = self.relu5(x11)
        x13 = self.conv6(x12)
        x14 = self.relu6(x13)
        x15 = self.maxpool3(x14)

        x16 = self.conv7(x15)
        x17 = self.relu7(x16)
        x18 = self.conv8(x17)
        x19 = self.relu8(x18)
        # reshape to [1, 512, 70, 70]

        x20 = self.maxpool4(x19)
        print(x19.shape)

        x21 = self.conv9(x20)
        x22 = self.relu9(x21)
        x23 = self.conv10(x22)
        x24 = self.relu10(x23)

        # Decoder

        x25 = self.trans1(x24)
        # concatenate x19 and x25
        cat = torch.cat([x19, x25], dim=1)
        x26 = self.conv11(cat)
        x27 = self.relu11(x26)
        x28 = self.conv12(x27)
        x29 = self.relu12(x28)

        x30 = self.trans2(x29)
        cat2 = torch.cat([x14, x30], dim=1)
        x31 = self.conv13(cat2)
        x32 = self.relu13(x31)
        x33 = self.conv14(x32)
        x34 = self.trans3(x33)
        cat3 = torch.cat([x9, x34], dim=1)
        x35 = self.conv15(cat3)
        x36 = self.relu15(x35)
        x37 = self.conv16(x36)
        x38 = self.trans4(x37)
        cat4 = torch.cat([x4, x38], dim=1)
        x39 = self.conv17(cat4)
        x40 = self.relu17(x39)
        x41 = self.conv18(x40)
        x42 = self.conv19(x41)
        x42 = self.sigmoid(x42)

        return x42


from torch.nn.functional import relu


class UNet1(nn.Module):
    """
    unet model




    """

    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out


if __name__ == "__main__":
    model = Unet()
    x = torch.randn(1, 3, 256, 256)
    print(model)

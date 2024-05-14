import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        out_channels_per_layer,
        kernel_sizes_per_layer,
        maxpool_per_layer,
    ):
        super().__init__()

        self.out_channels_per_layer = out_channels_per_layer
        self.kernel_sizes_per_layer = kernel_sizes_per_layer
        self.maxpool_per_layer = maxpool_per_layer

        in_features, h, w = input_shape
        layers = nn.ModuleList()

        for i in range(len(out_channels_per_layer)):
            block = [
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_channels_per_layer[i],
                    kernel_size=kernel_sizes_per_layer[i],
                    padding='same',
                ),
                nn.ReLU(),
            ]
            if maxpool_per_layer[i]:
                block.append(nn.MaxPool2d(kernel_size=maxpool_per_layer[i]))
                h //= maxpool_per_layer[i]
                w //= maxpool_per_layer[i]

            layers.append(nn.Sequential(*block))
            in_features = out_channels_per_layer[i]
        
        self.input_shape = input_shape
        self.output_shape = (out_channels_per_layer[-1], h, w)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

class CNNDecoder(nn.Module):
    def __init__(
        self,
        input_shape,
        out_channels_per_layer,
        kernel_sizes_per_layer,
        upsample_per_layer,
    ):
        super().__init__()

        self.out_channels_per_layer = out_channels_per_layer
        self.kernel_sizes_per_layer = kernel_sizes_per_layer
        self.upsample_per_layer = upsample_per_layer

        in_features, h, w = input_shape
        layers = nn.ModuleList()

        for i in range(len(out_channels_per_layer)):
            block = [
                nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_channels_per_layer[i],
                    kernel_size=kernel_sizes_per_layer[i],
                    padding='same',
                ),
                nn.ReLU(),
            ]
            if upsample_per_layer[i]:
                block.append(nn.Upsample(scale_factor=upsample_per_layer[i], mode='bicubic'))
                h *= upsample_per_layer[i]
                w *= upsample_per_layer[i]

            layers.append(nn.Sequential(*block))
            in_features = out_channels_per_layer[i]
        
        self.input_shape = input_shape
        self.output_shape = (in_features, h, w)

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class GenWeakSegNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.input_shape = (3, 224, 224)
        self.num_classes = num_classes

        out_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
        maxpool = [0, 2, 0, 2, 0, 2, 0, 2]

        self.encoder = CNNEncoder(
            input_shape=self.input_shape,
            out_channels_per_layer=out_channels,
            kernel_sizes_per_layer=kernel_sizes,
            maxpool_per_layer=maxpool,
        )

        out_channels = out_channels[::-1]
        kernel_sizes = kernel_sizes[::-1]
        upsample = maxpool

        self.decoder = CNNDecoder(
            input_shape=self.encoder.output_shape,
            out_channels_per_layer=out_channels,
            kernel_sizes_per_layer=kernel_sizes,
            upsample_per_layer=upsample,
        )

        self.op_cls_img = nn.Conv2d(out_channels[-1], 3*num_classes, kernel_size=3, padding='same')
        self.op_cls_mask = nn.Conv2d(out_channels[-1], num_classes, kernel_size=3, padding='same')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        y_img = self.op_cls_img(x).view(x.shape[0], self.num_classes, *self.input_shape)
        y_mask = self.op_cls_mask(x)
        return (y_img, y_mask)

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, image_shape, generator_filters):
        super(Generator, self).__init__()

        self.image_shape = image_shape
        self.generator_filters = generator_filters

        self.downsample_1 = self.conv2d(image_shape[0], generator_filters, bn=False)
        self.downsample_2 = self.conv2d(generator_filters, generator_filters*2)
        self.downsample_3 = self.conv2d(generator_filters*2, generator_filters*4)
        self.downsample_4 = self.conv2d(generator_filters*4, generator_filters*8)
        self.downsample_5 = self.conv2d(generator_filters*8, generator_filters*8)
        # self.downsample_6 = self.conv2d(generator_filters*8, generator_filters*8)
        # self.downsample_7 = self.conv2d(generator_filters*8, generator_filters*8)

        # self.upsample_1 = self.deconv2d(generator_filters*8, generator_filters*8)
        # self.upsample_2 = self.deconv2d(generator_filters*8, generator_filters*8)
        self.upsample_3 = self.deconv2d(generator_filters*8, generator_filters*8)
        self.upsample_4 = self.deconv2d(generator_filters*8, generator_filters*4)
        self.upsample_5 = self.deconv2d(generator_filters*4, generator_filters*2)
        self.upsample_6 = self.deconv2d(generator_filters*2, generator_filters)
        self.upsample_7 = self.deconv2d(generator_filters, image_shape[0], bn=False)  # Upsample to the final number of channels

    def conv2d(self, in_channels, filters, bn=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, filters, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if bn:
            layers.append(nn.BatchNorm2d(filters))
        return nn.Sequential(*layers)

    def deconv2d(self, in_channels, out_channels, dropout_rate=0, bn=True):
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if dropout_rate:
            layers.append(nn.Dropout(dropout_rate))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)



    def forward(self, x):
        downsample_1 = self.downsample_1(x)
        downsample_2 = self.downsample_2(downsample_1)
        downsample_3 = self.downsample_3(downsample_2)
        downsample_4 = self.downsample_4(downsample_3)
        downsample_5 = self.downsample_5(downsample_4)
        # downsample_6 = self.downsample_6(downsample_5)
        # downsample_7 = self.downsample_7(downsample_6)

        # upsample_1 = self.upsample_1(downsample_7)
        # upsample_2 = self.upsample_2(upsample_1)
        upsample_3 = self.upsample_3(downsample_5)
        upsample_4 = self.upsample_4(upsample_3)
        upsample_5 = self.upsample_5(upsample_4)
        upsample_6 = self.upsample_6(upsample_5)
        upsample_7 = self.upsample_7(upsample_6)

        return torch.tanh(upsample_7)

# Create an instance of the generator



import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_shape, discriminator_filters):
        super(Discriminator, self).__init__()

        self.image_shape = image_shape
        self.discriminator_filters = discriminator_filters

        # Define layers
        self.discriminator_layer_1 = self.discriminator_layer(image_shape[0]*2, discriminator_filters, bn=False)
        self.discriminator_layer_2 = self.discriminator_layer(discriminator_filters, discriminator_filters*2)
        self.discriminator_layer_3 = self.discriminator_layer(discriminator_filters*2, discriminator_filters*4)
        self.discriminator_layer_4 = self.discriminator_layer(discriminator_filters*4, discriminator_filters*8)
        self.validity = nn.Conv2d(discriminator_filters*8, 1, kernel_size=4, stride=1, padding=1)

    def discriminator_layer(self, in_channels, filters, bn=True):
        layers = []
        layers.append(nn.Conv2d(in_channels, filters, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if bn:
            layers.append(nn.BatchNorm2d(filters))
        return nn.Sequential(*layers)

    def forward(self, source_image, destination_image):
        # Concatenate along the channels axis
        combined_images = torch.cat((source_image, destination_image), dim=1)
        
        discriminator_layer_1 = self.discriminator_layer_1(combined_images)
        discriminator_layer_2 = self.discriminator_layer_2(discriminator_layer_1)
        discriminator_layer_3 = self.discriminator_layer_3(discriminator_layer_2)
        discriminator_layer_4 = self.discriminator_layer_4(discriminator_layer_3)
        validity = self.validity(discriminator_layer_4)
        return validity

# Create an instance of the discriminator

image_shape = (3, 256, 256)  # Assuming RGB images with size 256x256
discriminator_filters = 64
discriminator = Discriminator(image_shape, discriminator_filters)
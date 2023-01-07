import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            
            # Using a bias in the convolution layer will be cancelled by the batch norm
            nn.BatchNorm2d(out_channels)
            nn.ReLu(inplace=True)

            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            nn.BatchNorm2d(out_channels)
            nn.ReLu(inplace=True)
        )



    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()


        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            # self.downs.append(DoubleConv(in_channels=feature))
            # self.ups.append(DoubleConv(in_channels=feature))
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
                
        
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            ##????
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]


        # There are two different 'elements' in the 'up' stack
        # ConvTranspose2d and DoubleConv
        for idx in range(0, len(self.ups), 2):

            # Pass x to the ConvTranspose2d of self.ups at idx
            x = self.ups[idx](x)

            # Grab the 'correct' skip connection
            skip_connection = skip_connections[idx//2]

            # Concat the skip connection to x
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Pass x to the final DoubleConv of self.ups
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)







            


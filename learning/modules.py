import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def forward(self, data_labels, synthesis_labels):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        target = torch.ones(1).expand_as(synthesis_labels).to(device)
        loss = (data_labels.pow(2).sum() +
                (target - synthesis_labels).pow(2).sum()) /\
                (data_labels.numel() + synthesis_labels.numel())
        return loss

class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size, padding=0),
            nn.InstanceNorm2d(n_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.ReflectionPad2d(1),
            nn.Conv2d(n_channels, n_channels, kernel_size, padding=0),
            nn.InstanceNorm2d(n_channels),
        )

    def forward(self, x):
        return x + self.model(x)

class Generator(nn.Module):
    def __init__(self, n_residual_blocks):
        super(Generator, self).__init__()
        modules_down = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        ]
        modules_residuals = [
            ResidualBlock(256, 3)
            for i in range(n_residual_blocks)
        ]
        modules_up = [
            nn.ConvTranspose2d(256, 128, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, stride=1, padding=0),
            nn.Tanh()
        ]

        modules = modules_down + modules_residuals + modules_up
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)

    def sample(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch, _ = next(iter(data_loader))
        batch = batch.to(device)
        return self.forward(batch)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 256 x 256 x 3
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 128 x 128 x 64
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 64 x 64 x 128
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 32 x 32 x 256
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 16 x 16 x 512
            nn.Conv2d(512, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # 15 x 15 x 512
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
            # 14 x 14 x 1
        )

    def forward(self, x):
        return self.model(x)

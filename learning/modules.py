import torch.nn as nn
import utils


class CycleGANLoss(nn.Module):
    def __init__(self, regularizer, inverted, device):
        super(CycleGANLoss, self).__init__()
        self.regularizer = regularizer
        self.inverted = inverted
        self.device = device

    def forward(self, data):
        mse = nn.MSELoss(reduction='elementwise_mean').to(self.device)
        l1norm = nn.L1Loss(reduction='elementwise_mean').to(self.device)

        batch_targets = utils.get_target(
            True, self.inverted, data.batch_x_predictions.size(), self.device)
        synthesis_targets = utils.get_target(
            False, self.inverted, data.synthesis_x_predictions.size(),
            self.device)

        return mse(data.batch_x_predictions, batch_targets) +\
            mse(data.synthesis_x_predictions, synthesis_targets) +\
            mse(data.batch_y_predictions, batch_targets) +\
            mse(data.synthesis_y_predictions, synthesis_targets) +\
            self.regularizer * l1norm(data.id_x_approximations,
                                      data.batch_x) +\
            self.regularizer * l1norm(data.id_y_approximations,
                                      data.batch_y)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, use_dropout):
        super(ResidualBlock, self).__init__()
        block_1 = [nn.ReflectionPad2d(1),
                   nn.Conv2d(n_channels, n_channels, kernel_size, padding=0),
                   nn.InstanceNorm2d(n_channels),
                   nn.ReLU(inplace=True)]
        regularizer = [nn.Dropout(0.5)] if use_dropout else []
        block_2 = [nn.ReflectionPad2d(1),
                   nn.Conv2d(n_channels, n_channels, kernel_size, padding=0),
                   nn.InstanceNorm2d(n_channels)]
        modules = block_1 + regularizer + block_2
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return x + self.model(x)


class Generator(nn.Module):
    def __init__(self, n_residual_blocks, use_dropout):
        super(Generator, self).__init__()
        modules_down = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        ]
        modules_residuals = [
            ResidualBlock(
                256,
                3,
                use_dropout) for i in range(n_residual_blocks)]
        modules_up = [
            nn.ConvTranspose2d(256, 128, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2,
                               padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, stride=1, padding=0),
            nn.Tanh()
        ]

        modules = modules_down + modules_residuals + modules_up
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

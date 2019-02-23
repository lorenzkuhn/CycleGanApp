import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

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

def sample_from_data(data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch, _ = next(iter(data_loader))
    batch = batch.to(device)
    return batch

def get_labels(data_loader, generator, discriminator):
    data_samples = sample_from_data(data_loader)
    synthesis_samples = generator.sample(data_loader)
    data_labels = discriminator(data_samples)
    synthesis_labels = discriminator(synthesis_samples)
    return data_labels, synthesis_labels

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def log(loss):
    with open('log_new2.csv', 'a') as log_file:
        log_file.write(str(loss) + '\n')

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(7)
    train_data_path = 'train_A'
    n_residual_blocks = 9
    discriminator = Discriminator()
    discriminator = discriminator.to(device)
    generator = Generator(n_residual_blocks)
    generator = generator.to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.002)
    generator_optimizer = optim.Adam(generator.parameters(), lr=0.002)
    batchsize = 2
    n_epochs = 100
    n_discriminator_steps = 1
    image_size = (128, 128)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.ImageFolder(
        root=train_data_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batchsize, shuffle=True, num_workers=4)

    loss_function = nn.MSELoss(reduction='elementwise_mean').to(device)
    custom_loss_function = CustomLoss()
    custom_loss_function = custom_loss_function.to(device)
    for epoch_index in range(n_epochs):
        for _ in range(n_discriminator_steps):
            discriminator_optimizer.zero_grad()
            data_labels, synthesis_labels = get_labels(
                train_data_loader, generator, discriminator)

            # Maximizing loss function - hence inverting labels.
            data_targets = torch.ones(1).expand_as(data_labels).to(device)
            synthesis_targets = torch.zeros(1).expand_as(synthesis_labels).to(device)

            loss = loss_function(data_labels, data_targets) +\
                loss_function(synthesis_labels, synthesis_targets)

            loss.backward()

            discriminator_optimizer.step()

            custom_loss = custom_loss_function(data_labels, synthesis_labels)
            log(custom_loss.item())
        generator_optimizer.zero_grad()
        data_labels, synthesis_labels = get_labels(
            train_data_loader, generator, discriminator)

        synthesis_targets = torch.ones(1).expand_as(synthesis_labels).to(device)
        loss = loss_function(synthesis_labels, synthesis_targets)
        loss.backward()
        generator_optimizer.step()

        custom_loss = custom_loss_function(data_labels, synthesis_labels)
        log(custom_loss.item())
        # log(epoch_index)

    torch.save(generator.state_dict(), 'awsm_model_2')
    # dummy_input = torch.randn(batchsize, 3, 128, 128)
    # torch.onnx.export(generator, dummy_input, 'generator.onnx')

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from modules import CustomLoss, Discriminator, Generator

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

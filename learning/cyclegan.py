import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from modules import CycleGANLoss, Discriminator, Generator
import utils


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(7)
    n_residual_blocks = 9
    use_dropout = False
    path_train_data_x = 'train_A'
    path_train_data_y = 'train_B'
    path_model_xy = 'cyclegan_gen_AB'
    path_model_yx = 'cyclegan_gen_BA'
    batchsize = 5
    n_epochs = 250
    regularizer = 10
    n_discriminator_steps = 1
    image_size = (256, 256)

    discr_x = Discriminator()
    discr_x = discr_x.to(device)
    discr_y = Discriminator()
    discr_y = discr_y.to(device)
    gen_xy = Generator(n_residual_blocks, use_dropout)
    gen_xy = gen_xy.to(device)
    gen_yx = Generator(n_residual_blocks, use_dropout)
    gen_yx = gen_yx.to(device)
    discr_x.apply(utils.init_weights_gaussian)
    discr_y.apply(utils.init_weights_gaussian)
    gen_xy.apply(utils.init_weights_gaussian)
    gen_yx.apply(utils.init_weights_gaussian)

    optimizer_discr_x = optim.Adam(discr_x.parameters(), lr=0.002)
    optimizer_discr_y = optim.Adam(discr_y.parameters(), lr=0.002)
    optimizer_gen_xy = optim.Adam(gen_xy.parameters(), lr=0.002)
    optimizer_gen_yx = optim.Adam(gen_yx.parameters(), lr=0.002)

    transform = utils.get_transform(image_size)
    train_data_x = torchvision.datasets.ImageFolder(
        root=path_train_data_x, transform=transform)
    train_data_x_loader = torch.utils.data.DataLoader(
        train_data_x, batch_size=batchsize, shuffle=True, num_workers=4)
    train_data_y = torchvision.datasets.ImageFolder(
        root=path_train_data_y, transform=transform)
    train_data_y_loader = torch.utils.data.DataLoader(
        train_data_y, batch_size=batchsize, shuffle=True, num_workers=4)

    mse = nn.MSELoss(reduction='elementwise_mean').to(device)
    loss_function = CycleGANLoss(regularizer, False, device)

    for epoch_index in range(n_epochs):
        for _ in range(n_discriminator_steps):
            optimizer_discr_x.zero_grad()
            optimizer_discr_y.zero_grad()

            cycle_data = utils.get_batch_data(
                train_data_x_loader, train_data_y_loader, discr_x, discr_y,
                gen_xy, gen_yx, device)

            # Maximizing loss function - hence inverting labels.
            size = cycle_data.batch_x_predictions.size()
            batch_targets = utils.get_target(True, True, size, device)
            synthesis_targets = utils.get_target(False, True, size, device)

            loss_x = mse(cycle_data.batch_x_predictions, batch_targets) +\
                mse(cycle_data.synthesis_x_predictions, synthesis_targets)
            loss_y = mse(cycle_data.batch_y_predictions, batch_targets) +\
                mse(cycle_data.synthesis_y_predictions, synthesis_targets)

            loss_x.backward()
            optimizer_discr_x.step()
            loss_y.backward()
            optimizer_discr_y.step()

        optimizer_gen_xy.zero_grad()
        optimizer_gen_yx.zero_grad()
        cycle_data = utils.get_batch_data(
            train_data_x_loader, train_data_y_loader, discr_x, discr_y, gen_xy,
            gen_yx, device)
        loss = loss_function(cycle_data)

        loss.backward(retain_graph=True)
        optimizer_gen_xy.step()
        loss.backward()
        optimizer_gen_yx.step()

    torch.save(gen_xy.state_dict(), path_model_xy)
    torch.save(gen_yx.state_dict(), path_model_yx)


if __name__ == "__main__":
    train()

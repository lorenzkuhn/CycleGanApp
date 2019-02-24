import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from nn_modules import CycleGANLoss, Discriminator, Generator
import utils


def train():

    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--n_epochs', default=2, type=int)
    args = parser.parse_args()

    # batch_size = 1
    batch_size = args.batch_size
    # n_epochs = 250
    n_epochs = args.n_epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(7)
    n_residual_blocks = 9
    use_dropout = False
    path_train_data_x = 'trainA'
    path_train_data_y = 'trainB'
    path_model_xy = 'cyclegan_gen_AB'
    path_model_yx = 'cyclegan_gen_BA'
    # learning_rate = .002
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

    optimizer_discr_x = optim.Adam(discr_x.parameters())
    optimizer_discr_y = optim.Adam(discr_y.parameters())
    optimizer_gen_xy = optim.Adam(gen_xy.parameters())
    optimizer_gen_yx = optim.Adam(gen_yx.parameters())

    scheduler_discr_x = utils.HingeScheduler(optimizer_discr_x, .0001, 100, 100)
    scheduler_discr_y = utils.HingeScheduler(optimizer_discr_y, .0001, 100, 100)
    scheduler_gen_xy = utils.HingeScheduler(optimizer_gen_xy, .0002, 100, 100)
    scheduler_gen_yx = utils.HingeScheduler(optimizer_gen_yx, .0002, 100, 100)

    transform = utils.get_transform(image_size)
    train_data_x = torchvision.datasets.ImageFolder(
        root=path_train_data_x, transform=transform)
    train_data_x_loader = torch.utils.data.DataLoader(
        train_data_x, batch_size=batch_size, shuffle=True, num_workers=4)
    train_data_y = torchvision.datasets.ImageFolder(
        root=path_train_data_y, transform=transform)
    train_data_y_loader = torch.utils.data.DataLoader(
        train_data_y, batch_size=batch_size, shuffle=True, num_workers=4)

    synthesis_x_pool = utils.HistoricPool(50)
    synthesis_y_pool = utils.HistoricPool(50)

    mse = nn.MSELoss(reduction='mean').to(device)
    loss_function = CycleGANLoss(regularizer, False, device)

    for epoch_index in range(n_epochs):
        for _ in range(n_discriminator_steps):
            utils.switch_cycle_gradient_requirements(
                discr_x, discr_y, gen_xy, gen_yx, True)
            optimizer_discr_x.zero_grad()
            optimizer_discr_y.zero_grad()
            scheduler_discr_x.step()
            scheduler_discr_y.step()

            cycle_data = utils.get_cycle_data(
                train_data_x_loader, train_data_y_loader, discr_x, discr_y,
                gen_xy, gen_yx, device)

            synthesis_x_pool.update(cycle_data.synthesis_x)
            synthesis_y_pool.update(cycle_data.synthesis_y)
            synthesis_x_batch = synthesis_x_pool.get_batch()
            synthesis_y_batch = synthesis_y_pool.get_batch()

            # Maximizing loss function - hence inverting labels.
            size = cycle_data.real_x_predictions.size()
            real_targets = utils.get_target(True, True, size, device)
            synthesis_targets = utils.get_target(False, True, size, device)

            loss_x = mse(cycle_data.real_x_predictions, real_targets) +\
                mse(discr_x(synthesis_x_batch), synthesis_targets)
            loss_x.backward()
            optimizer_discr_x.step()

            loss_y = mse(cycle_data.real_y_predictions, real_targets) +\
                mse(discr_y(synthesis_y_batch), synthesis_targets)
            loss_y.backward()
            optimizer_discr_y.step()

        utils.switch_cycle_gradient_requirements(
            discr_x, discr_y, gen_xy, gen_yx, False)

        optimizer_gen_xy.zero_grad()
        optimizer_gen_yx.zero_grad()
        scheduler_gen_xy.step()
        scheduler_gen_yx.step()
        cycle_data = utils.get_cycle_data(
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

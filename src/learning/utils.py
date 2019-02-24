import random
import torch
import torchvision.transforms as transforms


class HistoricPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.pool = []
        self.next_batch = []

    def update(self, new_images):
        # If len(pool) + len(images) < pool_size, the next batch will consist
        # of purely new images. Whether this is desirable is very debatable.
        self.next_batch = []
        for new_image_index in range(new_images.size()[0]):
            new_image = new_images[new_image_index].unsqueeze(0)
            if len(self.pool) < self.pool_size:
                self.pool.append(new_image)
                self.next_batch.append(new_image)
            else:
                if random.uniform(0, 1) > .5:
                    self.next_batch.append(new_image)
                else:
                    pool_sample_index = random.randint(
                        0, self.pool_size - 1)
                    self.next_batch.append(self.pool[pool_sample_index])
                    self.pool[pool_sample_index] = new_image

    def get_batch(self):
        return torch.cat(self.next_batch, dim=0)


class HingeScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_lr, n_max_epochs, n_decayed_epochs,
                 last_epoch=-1):
        self.max_lrs = [max_lr for group in optimizer.param_groups]
        self.n_max_epochs = n_max_epochs
        self.n_decayed_epochs = n_decayed_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.n_max_epochs:
            return self.max_lrs
        return [lr * (self.last_epoch - self.n_max_epochs)
                / self.n_decayed_epochs
                for lr in self.max_lrs]


class CycleData():
    def __init__(self, discr_x, discr_y, gen_xy, gen_yx, real_x, real_y):
        self.real_x = real_x
        self.real_y = real_y
        self.synthesis_x = gen_yx(real_y)
        self.synthesis_y = gen_xy(real_x)
        self.real_x_predictions = discr_x(self.real_x)
        self.real_y_predictions = discr_y(self.real_y)
        self.synthesis_x_predictions = discr_x(self.synthesis_x)
        self.synthesis_y_predictions = discr_y(self.synthesis_y)
        self.id_x_approximations = gen_yx(self.synthesis_y)
        self.id_y_approximations = gen_xy(self.synthesis_x)


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_cycle_data(data_x_loader, data_y_loader, discr_x, discr_y, gen_xy,
                   gen_yx, device):
    real_x, _ = next(iter(data_x_loader))
    real_x = real_x.to(device)
    real_y, _ = next(iter(data_y_loader))
    real_y = real_y.to(device)
    cycle_data = CycleData(discr_x, discr_y, gen_xy, gen_yx, real_x, real_y)
    return cycle_data


def init_weights_gaussian(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)


def get_target(real, inverted, shape, device):
    if (real and not inverted) or (not real and inverted):
        return torch.zeros(shape).to(device)
    return torch.ones(shape).to(device)


def write_to_loss_logging(loss):
    with open('loss.csv', 'a') as loss_file:
        loss_file.write(str(loss) + '\n')

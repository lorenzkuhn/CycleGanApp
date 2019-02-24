import torch
import torchvision
from torchvision.utils import save_image
from nn_modules import Generator
from utils import get_transform

MODEL_PATH_XY = 'cyclegan_gen_AB'
MODEL_PATH_YX = 'cyclegan_gen_BA'

VALIDATION_DATA_PATH_X = 'testA'

DISPLAY_IMAGE = False
N_VALIDATION_IMAGES = 3
N_RESIDUAL_BLOCKS = 9
USE_DROPOUT = False
IMAGE_SIZE = (256, 256)


def main():
    transform = get_transform(IMAGE_SIZE)
    validation_data = torchvision.datasets.ImageFolder(
        root=VALIDATION_DATA_PATH_X, transform=transform)

    model_xy = Generator(N_RESIDUAL_BLOCKS, USE_DROPOUT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_xy.load_state_dict(torch.load(MODEL_PATH_XY, map_location=device))
    model_xy.eval()

    model_yx = Generator(N_RESIDUAL_BLOCKS, USE_DROPOUT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_yx.load_state_dict(torch.load(MODEL_PATH_YX, map_location=device))
    model_yx.eval()

    for i in range(N_VALIDATION_IMAGES):
        prediction = model_yx(model_xy(validation_data[i][0].unsqueeze(0)))
        save_image(prediction, f'cycle_inference{i}.png', normalize=True)


if __name__ == "__main__":
    main()

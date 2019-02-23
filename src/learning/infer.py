import torch
import torchvision
from torchvision.utils import save_image
from nn_modules import Generator
from utils import get_transform

MODEL_PATH = 'gpu_model'
VALIDATION_DATA_PATH = 'test_A'

DISPLAY_IMAGE = False
N_VALIDATION_IMAGES = 3
N_RESIDUAL_BLOCKS = 9
USE_DROPOUT = False
IMAGE_SIZE = (256, 256)


def main():
    model = Generator(N_RESIDUAL_BLOCKS, USE_DROPOUT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    transform = get_transform(IMAGE_SIZE)
    validation_data = torchvision.datasets.ImageFolder(
        root=VALIDATION_DATA_PATH, transform=transform)

    for i in range(N_VALIDATION_IMAGES):
        prediction = model(validation_data[i][0].unsqueeze(0))
        save_image(prediction, f'prediction{i}.png', normalize=True)


if __name__ == "__main__":
    main()

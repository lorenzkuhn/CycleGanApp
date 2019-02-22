import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import gan

MODEL_PATH = 'awsm_model'
VALIDATION_DATA_PATH = 'test_A'

SAVE_IMAGE = True
DISPLAY_IMAGE = False
N_VALIDATION_IMAGES = 3

model = gan.Generator(9)
model.load_state_dict(torch.load(MODEL_PATH))
image_size = (128, 128)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
validation_data = torchvision.datasets.ImageFolder(
    root=VALIDATION_DATA_PATH, transform=transform)

for i in range(N_VALIDATION_IMAGES):

    prediction = model(validation_data[i][0].unsqueeze(0))
    if SAVE_IMAGE:
        save_image(prediction, 'prediction%d.png' % i, normalize=True)

    if DISPLAY_IMAGE:
        np_image = np.squeeze(prediction.data.numpy(), axis=0)
        # Adapt to matplotlib dimensionality ordering.
        np_image = np.transpose(np_image, (1, 2, 0))
        for j in range(3):
            min_value = np.min(np_image[:, :, j])
            max_value = np.max(np_image[:, :, j])
            if min_value == max_value:
                print('channel: ' + str(j) + ' ; min: ' + str(min_value) +
                      ' ; max: ' + str(max_value))
                np_image[:, :, j] = .5
            np_image[:, :, j] = (np_image[:, :, j] - min_value) /\
                (max_value - min_value)
        plt.imshow(np_image)
        plt.show()

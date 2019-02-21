import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import gan

MODEL_PATH = 'awsm_model'
VALIDATION_DATA_PATH = 'test_A'

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

for i in range(3):
    # Add missing batch-dimensionality.
    image = validation_data[i][0].unsqueeze(0)
    inferred_image = np.squeeze(model(image).data.numpy(), axis=0)
    # Adapt to matplotlib dimensionality ordering.
    inferred_image = np.transpose(inferred_image, (1, 2, 0))

    for j in range(3):
        min_value = np.min(inferred_image[:, :, j])
        max_value = np.max(inferred_image[:, :, j])
        if min_value == max_value:
            print('channel: ' + str(j) + ' ; min: ' + str(min_value) +
                  ' ; max: ' + str(max_value))
            inferred_image[:, :, j] = .5
        inferred_image[:, :, j] = (inferred_image[:, :, j] - min_value) /\
            (max_value - min_value)

    plt.imshow(inferred_image)
    plt.show()

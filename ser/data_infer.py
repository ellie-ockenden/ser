from ser.data import test_dataloader
from ser.transforms import transforms, normalize


def data_load(label):
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))
    return images
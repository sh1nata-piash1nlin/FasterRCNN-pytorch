from torchvision.datasets import VOCDetection
from torchvision.transforms import Compose, ToTensor, Normalize
from pprint import pprint
import torch

class VOCDataset(VOCDetection):
    def __init__(self, root, year, image_set, download, transform):
        super().__init__(root, year, image_set, download, transform)
        self.categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                           'train', 'tvmonitor']

    def __getitem__(self, item):
        image, data = super().__getitem__(item)
        #print(image.s hape)
        all_bboxes = []
        all_lbls = []
        for obj in data["annotation"]["object"]:
            xmin = int(obj["bndbox"]["xmin"])
            ymin = int(obj["bndbox"]["ymin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymax = int(obj["bndbox"]["ymax"])
            all_bboxes.append([xmin, ymin, xmax, ymax])
            cat = obj["name"]     #label
            all_lbls.append(self.categories.index(cat))

        all_bboxes = torch.FloatTensor(all_bboxes) #ep ve tensor
        all_lbls = torch.LongTensor(all_lbls)
        #print(all_bboxes.shape)
        #print(all_bboxes)
        #print(all_lbls)
        target = {
            "boxes": all_bboxes,
            "labels": all_lbls,
        }

        return image, target


if __name__ == '__main__':
    transform = ToTensor()
    dataset = VOCDataset(root="../data/Pascal", year="2012", image_set="train", download=False, transform=transform)
    image, target = dataset[2000]
    # pprint(target["annotation"]["object"])
    # image.show()
    print(image.shape)
    print(target)
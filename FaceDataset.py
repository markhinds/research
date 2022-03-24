import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FaceDataset(Dataset):
    def __init__(self, root_dir, set, component, transform):
        self.root_dir = root_dir
        self.set = set
        self.component = component
        self.transform = transform

        self.total_imgs, self.total_values = self.get_data_list()
        self.total_values = np.array(self.total_values).astype(np.float)
        self.gt_mean = np.mean(self.total_values)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        try:
            image = Image.open(open(img_loc, 'rb')).convert("RGB")
        except:
            print(img_loc, 'was not loaded')

        # Normalize the image data
        image = self.transform(image)
        image = torch.tensor((image.numpy() - np.mean(image.numpy())) / np.std(image.numpy()))

        value = self.total_values[idx]

        return image, value

    def get_data_list(self):
        image_list, value_list = [], []
        img_path = self.root_dir + 'Data/Image/'
        anno_path = self.root_dir + 'Data/Annotation/' + self.set + self.component

        print('Anno file: ', anno_path)

        f = open(anno_path,'r')
        for i,line in enumerate(f):
            splitted_line = line.split(',')
            image_list.append(img_path+splitted_line[0])
            value_list.append(splitted_line[1])
        f.close()

        print(len(image_list), len(value_list))

        return image_list, value_list
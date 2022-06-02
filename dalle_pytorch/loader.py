from pathlib import Path
from random import sample

import torch
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
import random
import albumentations
import albumentations.pytorch

class MnistDataset(Dataset):
    def __init__(self,
                 text_folder,
                 img_folder,
                 image_size=64,
                 tokenizer=None,
                 split='train',
                 info_save_dir=None,
                 use_ratio=1.0,
                 ):
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.split = split
        text_path = Path(text_folder)
        img_path = Path(img_folder)

        print('data loading...')
        text_files = [*text_path.glob('**/*.txt')]
        image_files = [*img_path.glob('**/*.png')]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.tokenizer = tokenizer

        rescaler = albumentations.SmallestMaxSize(max_size = image_size)
        cropper = albumentations.RandomCrop(height=image_size, width=image_size)
        totensor = albumentations.pytorch.transforms.ToTensorV2() # The numpy HWC image is converted to pytorch CHW tensor.
        self.image_transform = albumentations.Compose([
            rescaler,
            cropper,
            totensor,
        ])

        self.keys = sorted(self.keys)
        random.Random(42).shuffle(self.keys)
        self.keys = self.keys[:int(use_ratio*len(self.keys))]

        # For consistent val/test split
        if self.split in ['val', 'test']:
            total_len = len(self.keys)
        
            with open(info_save_dir+'/val.txt', 'w') as f:
                for file_name in self.keys[:int(0.5*total_len)]:
                    f.write(file_name+'\n')

            with open(info_save_dir+'/test.txt', 'w') as f:
                for file_name in self.keys[int(0.5*total_len):]:
                    f.write(file_name+'\n')
            
            if split == 'val':
                self.keys = self.keys[:int(0.5*total_len)]
            elif split == 'test':
                self.keys = self.keys[int(0.5*total_len):]
        
    def __len__(self):
        return len(self.keys)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = (image/255.0).astype(np.float32)
        return image  # 0. ~ 1.

    def __getitem__(self, ind):
        key = self.keys[ind]

        text_file = self.text_files[key]   # Path
        image_file = self.image_files[key] # Path

        descriptions = text_file.read_text().split('\n')
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))

        if self.split == 'train':
            description = sample(descriptions, 1)  # sample: returns multiple random elements from the list without replacement.
            description = ' '.join(description)
        else:
            description = descriptions[0]

        tokenized_text_ids = self.tokenizer.encode(description).ids

        image = self.preprocess_image(image_file)
        image_tensor = self.image_transform(image=image)["image"]

        # Success
        return torch.tensor(tokenized_text_ids), image_tensor
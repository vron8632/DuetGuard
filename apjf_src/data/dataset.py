"""
Joint dataset for active-passive image forensic training.
- Loads images, generates synthetic tampered versions
- Creates watermark messages, secret images, SPN references
- 1:1 authentic/tampered sampling for balanced fusion training
"""
import torch
from torch.utils.data import Dataset
import random
import os
import numpy as np
from PIL import Image
from torchvision import transforms as T


class JointTrainDataset(Dataset):
    def __init__(self, image_dir, img_size=256, num_samples=2000,
                 tamper_ratio=0.5):
        self.img_size = img_size
        self.tamper_ratio = tamper_ratio
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

        all_images = []
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(image_dir, f))
        random.seed(42)
        random.shuffle(all_images)
        self.paths = all_images[:min(num_samples * 2, len(all_images))]

    def __getitem__(self, idx):
        img = self._load_image(self.paths[idx % len(self.paths)])

        is_tampered = (idx % 2 == 1) if idx < len(self.paths) else (random.random() < self.tamper_ratio)

        if is_tampered:
            idx2 = (idx + len(self.paths) // 2) % len(self.paths)
            other = self._load_image(self.paths[idx2])
            tamper_mask = self._random_mask()
            img = img * (1 - tamper_mask) + other * tamper_mask
            verdict = 2  # confirmed forgery
        else:
            tamper_mask = torch.zeros(1, self.img_size, self.img_size)
            verdict = 0  # authentic

        secret = torch.ones_like(img) * 0.5

        return {
            'image': img,
            'secret': secret,
            'tamper_mask': tamper_mask,
            'verdict': torch.tensor(verdict, dtype=torch.long),
            'is_tampered': torch.tensor(is_tampered, dtype=torch.float),
        }

    def _random_mask(self):
        mask = torch.zeros(1, self.img_size, self.img_size)
        x = random.randint(0, self.img_size // 2)
        y = random.randint(0, self.img_size // 2)
        w = random.randint(self.img_size // 4, self.img_size // 2)
        h = random.randint(self.img_size // 4, self.img_size // 2)
        mask[:, y:y+h, x:x+w] = 1.0
        return mask

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return max(len(self.paths), 200)

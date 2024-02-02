import os
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader

# 透過PNG画像の前処理関数
def preprocess_transparent_image(image_path, fill_color=(255, 255, 255)):
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, fill_color)
            background.paste(img, img.split()[-1])
            img = background
        return img

# 前処理済みの画像を保存する関数
def save_processed_image(image, original_path, save_dir='processed_images'):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(original_path))
    image.save(save_path)

# カスタムデータセットクラス
class CustomDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, save_dir='processed_images'):
        super(CustomDataset, self).__init__(root, transform)
        self.save_dir = save_dir

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = preprocess_transparent_image(path)
        save_processed_image(sample, path, self.save_dir)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# データセットとデータローダーの設定
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
train_dataset = CustomDataset(root='dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

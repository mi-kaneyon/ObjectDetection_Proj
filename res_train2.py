
from torchvision.transforms import RandomHorizontalFlip, Resize, RandomCrop, RandomRotation
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image
import json
import os
from engine import train_one_epoch, evaluate
import utils 
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def show_image_with_boxes(image, prediction, threshold=0.5):
    """画像と予測結果を元にバウンディングボックスを描画"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        if score > threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(box[0], box[1], f'{label}: {score:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def visualize_sample_images(model, dataset, device, num_samples=5):
    """指定された数のサンプル画像をモデルで予測し、結果を表示"""
    model.eval()  # モデルを評価モードに設定
    with torch.no_grad():
        for i in range(num_samples):
            img, _ = dataset[i]
            img = img.to(device)
            prediction = model([img])
            show_image_with_boxes(Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()), prediction[0])

# データ拡張の定義
def get_transform(train):
    transforms_list = [
        Resize((800, 800)),
        RandomHorizontalFlip(),
        RandomCrop((600, 600)),
        RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if train:
        transforms_list.append(ColorJitter(brightness=0.5, hue=0.3))
    return transforms.Compose(transforms_list)
    transforms_list = [transforms.ToTensor()]
    if train:
        transforms_list.append(transforms.ColorJitter(brightness=0.5, hue=0.3))
    return transforms.Compose(transforms_list)

# データセットクラス定義
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(annotations_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations[idx]['filename'])
        image = Image.open(img_path).convert("RGB")
        boxes = torch.as_tensor(self.annotations[idx]['bbox'], dtype=torch.float32)
        labels = torch.ones((1,), dtype=torch.int64)  # Assuming all instances have the same label
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

# モデル取得関数定義
def get_model(num_classes):
    # resnet50をキーワード引数とともに使用し、
    # 'pretrained'の代わりに'weights'引数を利用する
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights='IMAGENET1K_V1')
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )
    # Faster R-CNNの分類器の入力特徴量を取得
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 新しい分類器をモデルに設定
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
    
def main():
    # データセットのパスとアノテーションファイル
    annotations_file = 'annotations.json'
    img_dir = 'data/sekira/ok'
    num_classes = 2  # 1 class + background

    # データセットとデータローダーの設定
    dataset = CustomDataset(annotations_file, img_dir, transform=get_transform(train=True))
    dataset_test = CustomDataset(annotations_file, img_dir, transform=get_transform(train=False))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # モデルとオプティマイザーの設定
    model = get_model(num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    num_epochs = 10  # この行はlr_schedulerの前に移動する必要があります
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
    
    # トレーニングループ
    for epoch in range(num_epochs):
        # トレーニングフェーズ
        model.train()
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
        lr_scheduler.step()
        
        # 評価フェーズ
        evaluate(model, data_loader_test, device=device)

    # モデルの保存
    torch.save(model.state_dict(), 'res.pth')

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main()

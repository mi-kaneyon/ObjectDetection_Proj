import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import json
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou



# アノテーションファイルを読み込む
def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations




# データセットクラス
class SekiraDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = load_annotations(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_info = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_info["filename"])
        image = Image.open(img_path).convert("RGB")
        bbox = img_info["bbox"]
        boxes = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)  # Ensure it's always 2D
        labels = torch.tensor([1], dtype=torch.int64)  # Assuming 'ok' class index as 1

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target




def collate_fn(batch):
    images, targets = zip(*batch)
    # デバッグ出力を追加してtargetsの構造を確認
    print("Targets structure before processing:", targets)
    images = list(images)
    targets = [{k: v if isinstance(v, torch.Tensor) else torch.tensor(v) for k, v in t.items()} for t in targets]
    print("Targets structure after processing:", targets)
    return images, targets



# データ変換
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# データセットのインスタンス化
dataset = SekiraDataset(annotations_file='annotations.json', img_dir='data/sekira/ok', transform=transform)

# データローダーの定義
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# モデルの定義
model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# デバッグ情報の出力とデータローダーのイテレーションを開始
print("Starting data loading...")
for images, targets in dataloader:
    # targetsの構造を確認するデバッグ出力をここに追加
    print("Example target:", targets[0])
    print("Loaded batch")
    # 以降の処理...

# TensorBoardのログディレクトリを設定
writer = SummaryWriter('runs/ssd300_training')
# 最適化アルゴリズムと学習率スケジューラーの定義
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# トレーニングループ
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # トレーニングモードを確認
    running_loss = 0.0
    iou_scores = []  # IoUスコアのリストを初期化
    for images, targets in dataloader:
        images = [image.to(torch.device("cuda")) for image in images]
        targets = [{k: v.to(torch.device("cuda")) if hasattr(v, 'to') else v for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)  # ここでターゲットを含める
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        losses.backward()
        optimizer.step()

        # 推論用に評価モードに切り替え
        model.eval()
        with torch.no_grad():
            preds = model(images)  # 推論時にはターゲットを含めない
        
        # 予測されたバウンディングボックスと正解のバウンディングボックスのIoUを計算
        for idx, pred in enumerate(preds):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            true_boxes = targets[idx]['boxes']

            # スコアが0.5以上の予測を選択
            high_scores_idxs = pred_scores > 0.5
            pred_boxes_filtered = pred_boxes[high_scores_idxs]
            true_boxes_filtered = true_boxes

            # IoUを計算
            if pred_boxes_filtered.size(0) > 0 and true_boxes_filtered.size(0) > 0:  # Ensure there are boxes to compare
                iou = box_iou(pred_boxes_filtered, true_boxes_filtered).diag().mean().item()
                iou_scores.append(iou)

        model.train()  # 推論後にトレーニングモードに戻す

    # 平均IoUを計算
    average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0

    # 進捗を出力
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(dataloader)}, Average IoU: {average_iou}')

    # 学習率スケジューラーのステップを進める
    scheduler.step()

# モデルの保存
torch.save(model.state_dict(), 'vgg16.pth')

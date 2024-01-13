import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50

# SimpleLSTMクラスの定義
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# LSTMモデルのインスタンス化
lstm_model = SimpleLSTM(input_size=2, hidden_size=128, output_size=10)
lstm_model.eval()

# ResNet50モデルのロード
resnet_model = resnet50(pretrained=True)
resnet_model.eval()

# トランスフォームの定義
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# カメラの起動
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 動体検出用の背景減算器
backSub = cv2.createBackgroundSubtractorMOG2()

# 動体トラッキング用のデータ
timeseries_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgMask = backSub.apply(frame)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = frame[y:y+h, x:x+w]
            roi = transform(roi).unsqueeze(0)

            with torch.no_grad():
                resnet_output = resnet_model(roi)
                # ResNetの出力を解釈（省略）

            center_point = np.array([x + w/2, y + h/2])
            timeseries_data.append(center_point)

    if len(timeseries_data) > 0:
        lstm_input = torch.tensor(timeseries_data, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            lstm_output = lstm_model(lstm_input)
            probabilities = torch.softmax(lstm_output, dim=1)
            top_p, top_class = probabilities.topk(1, dim=1)
            label = top_class.item()
            score = top_p.item()
            cv2.putText(frame, f'Label: {label}, Score: {score:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

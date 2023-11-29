import torch

def load_and_print_info(model_path):
    # モデルの重みをロード
    state_dict = torch.load(model_path)

    print("Model's state_dict:")
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())

    # モデルの重みの形状とその他の情報を表示
    print("\nKeys in state_dict:")
    for key in state_dict.keys():
        print(key)

# モデルの.pthファイルへのパス
model_path = 'save_model.pth'  # ここに.pthファイルのパスを入力

# 情報の表示
load_and_print_info(model_path)

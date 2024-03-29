import torch
import numpy as np

def load_extended_statistics(filename='model_with_stats.pth'):
    """
    'model_with_stats.pth'から拡張統計情報を読み込みます。
    """
    # ファイルからチェックポイントをロード
    checkpoint = torch.load(filename, map_location='cpu')
    
    # 必要な統計情報の読み込み
    mean_vector = checkpoint.get('mean_vector')
    cov_matrix = checkpoint.get('cov_matrix')
    
    # モデルの最終層の出力形状に関する情報
    output_shape = checkpoint.get('output_shape')
    
    if mean_vector is not None and cov_matrix is not None and output_shape is not None:
        print("統計情報とモデルの出力形状を正常に読み込みました。")
        print("モデルの出力形状:", output_shape)
    else:
        print("必要な統計情報が見つかりませんでした。")
    
    return mean_vector, cov_matrix, output_shape

def print_extended_statistics(mean_vector, cov_matrix, output_shape):
    """
    平均ベクトル、共分散行列、およびモデルの出力形状の内容を表示します。
    """
    print("平均ベクトル:", mean_vector)
    print("共分散行列の形状:", cov_matrix.shape)
    print("モデルの出力形状:", output_shape)

if __name__ == '__main__':
    filename = 'model_with_stats.pth'  # 適切なファイルパスに置き換えてください
    mean_vector, cov_matrix, output_shape = load_extended_statistics(filename)
    print_extended_statistics(mean_vector, cov_matrix, output_shape)

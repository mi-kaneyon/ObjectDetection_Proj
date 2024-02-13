import torch
import argparse

def load_model_state_dict(file_path):
    try:
        state_dict = torch.load(file_path, map_location=torch.device('cpu'))
        return state_dict
    except Exception as e:
        print(f"Error loading state dictionary from {file_path}: {e}")
        return None

def analyze_model_structure_from_state_dict(state_dict):
    layer_types = set()
    prefixes = set()
    for key in state_dict.keys():
        parts = key.split('.')
        if len(parts) > 1:
            prefixes.add(parts[0])
        if 'weight' in key or 'bias' in key:
            layer_type = key.split('.')[-2]
            layer_types.add(layer_type)
    
    return {
        'total_layers': len(state_dict) // 2,  # Assuming each layer has a weight and a bias
        'layer_types': list(layer_types),
        'prefixes': list(prefixes)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a PyTorch model file (.pth) structure.")
    parser.add_argument("path", type=str, help="Path to the .pth model file")
    args = parser.parse_args()
    
    state_dict = load_model_state_dict(args.path)
    if state_dict:
        model_summary = analyze_model_structure_from_state_dict(state_dict)
        print("Model Summary:", model_summary)
    else:
        print("Error loading state dictionary.")

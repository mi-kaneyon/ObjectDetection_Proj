import subprocess

def check_nvidia_driver():
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        # ここでNVIDIAドライババージョンを解析
        print("NVIDIA driver is OK.")
        return True
    except:
        print("NVIDIA driver is NOT OK.")
        return False

def check_cuda_version():
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT, text=True)
        # ここでCUDAバージョンを解析
        print("CUDA version is OK.")
        return True
    except:
        print("CUDA version is NOT OK.")
        return False

# 他の3つの診断関数をここに追加

def main():
    checks = [check_nvidia_driver, check_cuda_version]  # 他の診断関数を追加
    all_passed = True
    
    for check in checks:
        result = check()
        if not result:
            all_passed = False
    
    if all_passed:
        print("All checks passed.")
    else:
        print("Some checks failed.")
        print("Solution: ...")  # 解決策を示す

if __name__ == "__main__":
    main()

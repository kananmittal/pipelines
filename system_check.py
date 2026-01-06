
import os
import sys
import platform
import subprocess
import importlib
import paddle.inference

# --- UNIVERSAL PATCH START ---
# Monkey-patch paddle.inference.create_predictor to force config safety globally
print("🔧 Applying Universal Paddle Config Patch...")
_original_create_predictor = paddle.inference.create_predictor

def _patched_create_predictor(config):
    # Depending on version, config might be AnalysisConfig or Config
    # We try both methods loosely
    try:
        # print("   -> Intercepted Config: Disabling IR Optim & MKLDNN")
        if hasattr(config, "switch_ir_optim"):
            config.switch_ir_optim(False)
        if hasattr(config, "disable_mkldnn"):
            config.disable_mkldnn()
        if hasattr(config, "enable_mkldnn"):
            # Ensure it's not re-enabled
            pass 
    except Exception as e:
        print(f"   -> Patch Warning: {e}")
    
    return _original_create_predictor(config)

paddle.inference.create_predictor = _patched_create_predictor
print("✅ Universal Patch Applied")
# --- UNIVERSAL PATCH END ---

def check_cpu_flags():
    print("\n--- CPU Info ---")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    flags = []
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            for line in cpuinfo.split("\n"):
                if "flags" in line:
                    flags = line.split(":")[1].strip().split()
                    break
        elif platform.system() == "Darwin":
            # Mac sysctl
            cmd = subprocess.run(["sysctl", "-a"], capture_output=True, text=True)
            if cmd.returncode == 0:
                flags = [line.split(":")[0].split(".")[-1] for line in cmd.stdout.split("\n") if "hw.optional" in line and ": 1" in line]
    except Exception as e:
        print(f"Could not read CPU flags: {e}")
        
    print(f"Supported Flags: {', '.join([f for f in ['avx', 'avx2', 'avx512', 'avx512f', 'sse4_1', 'sse4_2'] if any(x in f for x in flags)])}")
    return flags

def check_library(name):
    print(f"\n--- Checking {name} ---")
    try:
        lib = importlib.import_module(name)
        print(f"✅ {name} imported successfully")
        if hasattr(lib, "__version__"):
            print(f"Version: {lib.__version__}")
        return lib
    except ImportError:
        print(f"❌ {name} NOT found")
    except Exception as e:
        print(f"❌ {name} error on import: {e}")
    return None

def main():
    print(f"Python Version: {sys.version}")
    check_cpu_flags()
    
    # Check Torch (GPU)
    torch = check_library("torch")
    if torch:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version (Torch): {torch.version.cuda}")
            print(f"CUDNN Version (Torch): {torch.backends.cudnn.version()}")
            
    # Check Paddle
    paddle = check_library("paddle")
    if paddle:
        print(f"Paddle Device: {paddle.device.get_device()}")
        # Check if we can switch device
        try:
            paddle.set_device("cpu")
            print("✅ Successfully set paddle to CPU")
        except Exception as e:
            print(f"❌ Failed to set paddle to CPU: {e}")

    # Check PaddleOCR
    check_library("paddleocr")
    
    # Minimal OCR Run
    print("\n--- Minimal PaddleOCR Test ---")
    try:
        # Replicate environment variables set in pipeline
        os.environ["FLAGS_use_gpu"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["DNNL_MAX_CPU_ISA"] = "AVX2"
        os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX2"
        
        from paddleocr import PaddleOCR
        # Replicate the fixed constructor
        ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False, ir_optim=False, show_log=False)
        print("✅ PaddleOCR Initialized Successfully (CPU Mode, No Optim)")
        
        print("\n--- Testing PPStructure (The Crashing Component) ---")
        try:
            from paddleocr import PPStructure
            # Intentionally using the exact flags from Pipeline 2
            struct = PPStructure(
                table=True, 
                ocr=True, 
                show_log=True, 
                layout=True, 
                enable_mkldnn=False, 
                ir_optim=False
            )
            print("✅ PPStructure Initialized Successfully")
            
            # Simple dummy inference to trigger the graph optimization pass
            import numpy as np
            dummy_img = np.zeros((500, 500, 3), dtype=np.uint8)
            _ = struct(dummy_img)
            print("✅ PPStructure Inference Successful")
            
        except Exception as e:
            print(f"❌ PPStructure Failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ PaddleOCR Test Failed: {e}")

if __name__ == "__main__":
    main()

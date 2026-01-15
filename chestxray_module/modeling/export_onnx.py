
"""
Purpose:
    Export model in ONNX format for framework-agnostic downstream application.
    
Note:
    Pytoch, ONNX and python3.10+ comptability is not stable.
    Recommended to run this script:
    
        conda create -n onnx_export python=3.10 -y &&
        conda activate onnx_export &&
        conda install -c conda-forge pytorch torchvision pandas onnx onnxscript onnxruntime monai -y &&
        conda install -c conda-forge opencv -y
    
    Run script from this newly created env


"""

import torch
from pathlib import Path
from chestxray_module.modeling.predict import build_model  

# Varaibles config
CHECKPOINT_PATH = "models/best_model.pt"
ONNX_PATH = "models/best_model.onnx"
NUM_CLASSES = 3
DEVICE = "cpu"   # export on CPU (recommended)

# 1. Load model
model = build_model(num_classes=NUM_CLASSES)
state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# 2. Dummy input (batch=1, C=3, H=224, W=224)
dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

# 3. Export
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={
        "image": {0: "batch"},
        "logits": {0: "batch"},
    },
)

print(f"[OK] ONNX model saved to: {ONNX_PATH}")

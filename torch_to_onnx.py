import kornia
import torch

# Load LOFTR model
loftr = kornia.feature.LoFTR(pretrained='outdoor')

# Dummy input for the model
img1_tensor = torch.rand(1, 1, 224, 224)
img2_tensor = torch.rand(1, 1, 224, 224)

dummy_input = {'data': {"image0": img1_tensor, "image1": img2_tensor}}

# Check model output
# matches = loftr(dummy_input['data'])
# print(matches)

# Export the model to ONNX format
onnx_path = "loft_outdoor.onnx"
torch.onnx.export(loftr, dummy_input, onnx_path, verbose=True)

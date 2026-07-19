import torch
import sys
import os
from safetensors.torch import load_file, save_file

def convert():
    input_path = sys.argv[1]
    file_name, file_ext = os.path.splitext(input_path)

    if file_ext.lower() == ".pth":
        output_path = file_name + ".safetensors"
        print(f"Converting {input_path} to {output_path}...")

        state_dict = torch.load(input_path, map_location="cpu", weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        save_file(state_dict, output_path)
        print("Done!")

    else file_ext.lower() == ".safetensors":
        output_path = file_name + ".pth"
        print(f"Converting {input_path} to {output_path}...")

        state_dict = load_file(input_path, device="cpu")
        torch.save(state_dict, output_path)
        print("Done!")

if __name__ == "__main__":
    convert()
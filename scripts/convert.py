import os
import torch
from transformers import AutoModelForCausalLM
from pathlib import Path

def convert_safetensors_to_pth(model_dir, output_dir):
    # Load the model from the directory with SafeTensors
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Define the output path for the PyTorch model (.pth)
    output_path = Path(output_dir) / "llama-2-7b-chat.pth"

    # Save the model's state_dict
    torch.save(model.state_dict(), output_path)

    print(f"Model has been converted and saved to {output_path}")

# Directory where your model's safetensors files are stored
model_dir = "/opt/scratch/yongwu/gpt-fast/yongwww/llama-2-7b-chat-hf"

# Directory where you want to save the .pth file
output_dir = "/opt/scratch/yongwu/gpt-fast/yongwww/llama-2-7b-chat-hf"

convert_safetensors_to_pth(model_dir, output_dir)

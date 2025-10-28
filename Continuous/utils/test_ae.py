import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
os.environ["AE"] = "/jhcnas5/chenzhixuan/checkpoints/GenHancer/ae.safetensors"
import sys

# Add the Continuous directory to the path
sys.path.append("/home/chenzhixuan/Workspace/GenHancer/Continuous")

from src.flux.util import load_ae

def load_image(image_path, size=224):
    """Load and preprocess image"""
    image = Image.open(image_path).convert('RGB')
    
    # Resize and center crop
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor, image


def normalize_image(image_tensor, mean=0.5, std=0.5):
    """Normalize image for VAE"""
    return (image_tensor - mean) / std


def denormalize_image(image_tensor, mean=0.5, std=0.5):
    """Denormalize image from VAE output"""
    return image_tensor * std + mean


def save_image_grid(original, reconstructed, save_path):
    """Save original and reconstructed images side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    original_np = original[0].permute(1, 2, 0).cpu().numpy()
    original_np = np.clip(original_np, 0, 1)
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Reconstructed image
    recon_np = reconstructed[0].permute(1, 2, 0).cpu().numpy()
    recon_np = np.clip(recon_np, 0, 1)
    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison to: {save_path}")


def test_ae_reconstruction(image_path, model_name="flux-dev", device="cuda"):
    """
    Test VAE autoencoder reconstruction
    
    Args:
        image_path: Path to input image
        model_name: Model name (e.g., "flux-dev")
        device: Device to run on
    """
    print(f"Loading image from: {image_path}")
    original_tensor, original_pil = load_image(image_path)
    print(f"Original image shape: {original_tensor.shape}")
    
    # Load VAE
    print(f"Loading VAE model: {model_name}")
    vae = load_ae(model_name, device=device)
    vae.to(device)
    vae.eval()
    
    # Normalize for VAE
    normalized_input = normalize_image(original_tensor, mean=0.5, std=0.5)
    normalized_input = normalized_input.to(device).to(torch.float32)
    
    print(f"Normalized input shape: {normalized_input.shape}")
    
    # Encode
    print("Encoding image...")
    with torch.no_grad():
        encoded = vae.encode(normalized_input)
        print(f"Encoded shape: {encoded.shape}")
        
        # Decode
        print("Decoding latent...")
        decoded = vae.decode(encoded)
        print(f"Decoded shape: {decoded.shape}")

    # Denormalize
    reconstructed = denormalize_image(decoded, mean=0.5, std=0.5)
    
    # Calculate metrics (ensure same device and dtype)
    mse = torch.nn.functional.mse_loss(
        original_tensor.to(reconstructed.device).float(),
        reconstructed.float()
    )
    print(f"\nReconstruction MSE: {mse.item():.6f}")
    
    # Save results
    output_dir = "/home/chenzhixuan/Workspace/GenHancer/Continuous/utils/test_ae_output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "reconstruction_comparison.png")
    save_image_grid(original_tensor, reconstructed, output_path)
    
    # Save individual images
    original_path = os.path.join(output_dir, "original.png")
    reconstructed_path = os.path.join(output_dir, "reconstructed.png")
    
    original_pil.save(original_path)
    
    # Clamp to [0, 1] before converting to PIL to avoid visualization noise
    recon_pil = transforms.ToPILImage()(reconstructed.clamp(0, 1)[0].cpu())
    recon_pil.save(reconstructed_path)
    
    print(f"Saved original to: {original_path}")
    print(f"Saved reconstructed to: {reconstructed_path}")
    
    return original_tensor, reconstructed, encoded


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test VAE autoencoder reconstruction")
    parser.add_argument("--image", type=str, default="/home/chenzhixuan/Workspace/GenHancer/Continuous/generated_images/test2.png", help="Path to input image")
    parser.add_argument("--model", type=str, default="flux-dev", help="Model name")
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use")
    
    args = parser.parse_args()
    
    test_ae_reconstruction(args.image, args.model, args.device)

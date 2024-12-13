import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

class CustomDataset(Dataset):
    def __init__(self, color_image_dir, depth_map_dir, camera_data_dir, feed_width=1080, feed_height=1920):
        self.color_image_dir = color_image_dir
        self.depth_map_dir = depth_map_dir
        self.camera_data_dir = camera_data_dir
        self.feed_width = feed_width
        self.feed_height = feed_height
        
        # Load color and depth image file names
        self.color_image_paths = sorted([os.path.join(color_image_dir, f) for f in os.listdir(color_image_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.depth_map_paths = sorted([os.path.join(depth_map_dir, f) for f in os.listdir(depth_map_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        
        # Ensure that we have the same number of color and depth images
        assert len(self.color_image_paths) == len(self.depth_map_paths), "Mismatch in number of color and depth images"

    def __len__(self):
        return len(self.color_image_paths)
    
    def load_intrinsics(self):
        original_width = 1920
        original_height = 1080

        fx = original_width / 2
        fy = original_height / 2
        cx = original_width / 2
        cy = original_height / 2

        fx *= (self.feed_width / original_width)
        fy *= (self.feed_height / original_height)
        cx *= (self.feed_width / original_width)
        cy *= (self.feed_height / original_height)

        K = np.array([[fx, 0, cx, 0],
                      [0, fy, cy, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return K
    
    def __getitem__(self, idx):
        try:
            color_image = cv2.imread(self.color_image_paths[idx])
            if color_image is None:
                raise FileNotFoundError(f"Color image not found at {self.color_image_paths[idx]}")

            # Ensure dimensions are divisible by patch_size
            height, width = color_image.shape[:2]
            # target_height = (height // 16) * 16
            # target_width = (width // 16) * 16
            target_height = 224
            target_width = 224
            color_image = cv2.resize(color_image, (target_width, target_height))
            color_image = torch.from_numpy(color_image).permute(2, 0, 1).float() / 255.0

            depth_map = cv2.imread(self.depth_map_paths[idx], cv2.IMREAD_UNCHANGED)
            if depth_map is None:
                raise FileNotFoundError(f"Depth map not found at {self.depth_map_paths[idx]}")
            depth_map = cv2.resize(depth_map, (target_width, target_height))
            depth_map = depth_map.astype(np.float32)
            depth_map = torch.from_numpy(depth_map).unsqueeze(0).float()

            color_image_name = os.path.basename(self.color_image_paths[idx])
            camera_data_file = os.path.join(
                self.camera_data_dir,
                color_image_name.replace('.png', '.json').replace('.jpg', '.json')
            )

            if not os.path.exists(camera_data_file):
                raise FileNotFoundError(f"Camera data file not found at {camera_data_file}")

            with open(camera_data_file, 'r') as f:
                camera_info = json.load(f)

            camera_tensor = torch.tensor([
                camera_info['x'], camera_info['y'], camera_info['z'],
                camera_info['roll'], camera_info['pitch'], camera_info['yaw']
            ]).float()

            intrinsics = self.load_intrinsics()
            intrinsics_tensor = torch.from_numpy(intrinsics).float()

            sample = {
                'color_image': color_image,
                'depth_map': depth_map,
                'camera_data': camera_tensor,
                'intrinsics': intrinsics_tensor
            }

            return sample
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise e



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        
        self.patch_embed = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=0.1)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        #print(f"Input shape: {x.shape}")  # Expected: (batch_size, channels, height, width)

        # Ensure dimensions are divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(f"Input dimensions must be divisible by patch_size. Got H={H}, W={W}, patch_size={self.patch_size}")

        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        #print(f"after reshape: {x.shape}") 
        x = x.permute(0, 2, 4, 1, 3, 5)
        print(f"after permute: {x.shape}")
        x = x.reshape(B, -1, self.pos_embed)
        print(f"Shape after patch flattening: {x.shape}")  # Expected: (batch_size, num_patches, patch_size^2 * channels)

        x = self.patch_embed(x)
        print(f"Shape after patch embedding: {x.shape}")  # Expected: (batch_size, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B,-1, -1)
        
        x = torch.cat((cls_tokens, x), dim=1)
        #print(f"Shape after patch cat: {x.shape}")

        num_tokens = x.size(1)
        if self.pos_embed.size(1) != num_tokens:
            self.pos_embed = nn.Parameter(
                self.pos_embed[:, :num_tokens, :].clone().detach()
            )

        x += self.pos_embed
        #print(f"Shape after patch pos_embed: {x.shape}")
        x = self.dropout(x)

        for layer in self.transformer:
            x = layer(x)

        x = self.norm(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x + residual

class FusionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.rcu = ResidualConvUnit(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.rcu(x)
        x = self.upsample(x)
        return x

class DPT(nn.Module):
    def __init__(self, img_size=224, num_classes=1, embed_dim=768, variant='base'):
        super(DPT, self).__init__()
        self.variant = variant
        if variant == 'base':
            self.backbone = VisionTransformer(img_size=img_size,patch_size=16, embed_dim=embed_dim, depth=12)
        elif variant == 'large':
            self.backbone = VisionTransformer(img_size=img_size,patch_size=16, embed_dim=1024, depth=24)
        elif variant == 'hybrid':
            self.resnet = resnet50(pretrained=True)
            self.backbone = VisionTransformer(img_size=img_size,patch_size=16, embed_dim=embed_dim, depth=12)
        
        self.reassemble_32 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.reassemble_16 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.reassemble_8 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.reassemble_4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        self.fusion_32_16 = FusionBlock(embed_dim, embed_dim)
        self.fusion_16_8 = FusionBlock(embed_dim, embed_dim)
        self.fusion_8_4 = FusionBlock(embed_dim, embed_dim)
        
        self.output_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
    
    def forward(self, x, intrinsics=None):
        if self.variant == 'hybrid':
            x = self.resnet.conv1(x)
            x = self.resnet.bn1(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            x = self.resnet.layer3(x)
            x = self.resnet.layer4(x)
            x = self.backbone(x)
        else:
            x = self.backbone(x)
        
        x = x[:, 1:, :]  # Remove class token
        B, N, D = x.shape
        H = W = int(N**0.5)
        x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)  # BxDxHxW
        
        x_32 = self.reassemble_32(x)
        x_16 = self.fusion_32_16(x_32)
        x_8 = self.fusion_16_8(x_16)
        x_4 = self.fusion_8_4(x_8)
       #print(f"After fusion_8_4: {x_4.shape}")
        
        out = self.output_head(x_4)
        #print(f"Output head: {out.shape}")

        # Upsample to target resolution
        out = F.interpolate(out, size=(224, 224), mode='bilinear', align_corners=False)
        return out

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        color_image = batch['color_image'].to(device)
        depth_map = batch['depth_map'].to(device)

        optimizer.zero_grad()
        outputs = model(color_image)
        # print(f"outputs: {outputs.shape} dmap: {depth_map.shape}")
        loss = criterion(outputs, depth_map)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            color_image = batch['color_image'].to(device)
            depth_map = batch['depth_map'].to(device)

            outputs = model(color_image)

            loss = criterion(outputs, depth_map)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_dataloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_dpt_model.pth")
            print("Model saved.")

import os
import matplotlib.pyplot as plt
import torch

def test_model(model_path, test_dataset, device, output_dir="test_results", num_images=10):
    """
    Load a saved model, test it on a subset of test images, and save the outputs.

    Args:
        model_path (str): Path to the saved model weights.
        test_dataset (Dataset): Dataset containing test images.
        device (torch.device): The device (CPU/GPU) to run the test.
        output_dir (str): Directory to save the test results.
        num_images (int): Number of test images to evaluate.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model architecture
    model = DPT(img_size=224).to(device)  # Ensure this matches your trained model's architecture

    # Load the saved model weights
    print(f"Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for idx in range(min(num_images, len(test_dataset))):
            sample = test_dataset[idx]  # Get a single sample
            color_image = sample['color_image'].unsqueeze(0).to(device)  # Add batch dimension
            depth_map = sample['depth_map'].to(device)  # Target depth map

            # Get model prediction
            predicted_depth = model(color_image)

            # Convert to numpy for saving
            color_image_np = color_image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            depth_map_np = depth_map.squeeze(0).cpu().numpy()
            predicted_depth_np = predicted_depth.squeeze(0).squeeze(0).cpu().numpy()

            # Save the images
            color_image_path = os.path.join(output_dir, f"image_{idx + 1}_color.png")
            ground_truth_path = os.path.join(output_dir, f"image_{idx + 1}_ground_truth_depth.png")
            predicted_depth_path = os.path.join(output_dir, f"image_{idx + 1}_predicted_depth.png")

            # Save color image
            plt.imsave(color_image_path, color_image_np)

            # Save ground truth depth
            plt.imsave(ground_truth_path, depth_map_np, cmap="viridis")

            # Save predicted depth
            plt.imsave(predicted_depth_path, predicted_depth_np, cmap="viridis")

            print(f"Saved results for Image {idx + 1}:")
            print(f"  Color image: {color_image_path}")
            print(f"  Ground truth depth: {ground_truth_path}")
            print(f"  Predicted depth: {predicted_depth_path}")



# Parameters
img_size = 224
batch_size = 8
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize dataset and dataloaders
color_image_dir = "train_data/color/Town01_Opt_120/ClearNoon/height20m/rgb"
depth_map_dir = "train_data/depth/Town01_Opt_120/ClearNoon/height20m/depth"
camera_data_dir = "train_data/color/Town01_Opt_120/ClearNoon/height20m/camera"

# Assuming CustomDataset is already defined
dataset = CustomDataset(color_image_dir, depth_map_dir, camera_data_dir)

# Calculate the size of train and validation datasets
train_size = int(0.8 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation

# Split the dataset
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Initialize DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Initialize model, loss, and optimizer
model = DPT(img_size=img_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the model
train_model(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs)

saved_model_path = "dpt_model_patch8.pth"  # Path to the saved model weights

# Call the test function
test_model(
    model_path=saved_model_path,
    test_dataset=val_dataset,
    device=device,
    output_dir="test_outputs_patch8",  # Directory to save test results
    num_images=10               # Number of test images to process
)



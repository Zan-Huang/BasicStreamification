import sys
import os
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur

# Add the parent directory of DPC to the Python path
# Add the parent directory of DPC to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the DPC/dpc directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dpc'))

# Add the DPC/utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../utils'))


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_3d_lc import UCF101_3d, HMDB51_3d, MovingMNISTDigits, MovingMNISTMotions
from model_3d_lc import *
import matplotlib.pyplot as plt
import imageio

from resnet_2d3d import * 

from select_backbone import select_resnet
from convrnn import ConvGRU

from resnet_2d3d import ResNet2d3d_two_stream 
from resnet_2d3d import BasicBlock2d, BasicBlock3d

__all__ = [
    'resnet9_2d3d_full', 'ResNet2d3d_full', 'resnet18_2d3d_full', 'resnet34_2d3d_full', 'resnet50_2d3d_full', 'resnet101_2d3d_full',
    'resnet152_2d3d_full', 'resnet200_2d3d_full', 'resnet18_2d3d_two_stream'
]

def load_checkpoint(model, checkpoint_path, device):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create a new state_dict without the 'backbone.' prefix
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' and 'backbone.' prefixes if they exist
            if k.startswith('module.backbone.'):
                new_state_dict[k[len('module.backbone.'):]] = v
            elif k.startswith('backbone.'):
                new_state_dict[k[len('backbone.'):]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return

class Backbone(nn.Module):
    def __init__(self, block, layers, head1_dim, head2_dim):
        super(Backbone, self).__init__()
        self.stream1 = ResNet2d3d_two_stream(block, layers).stream1
        self.stream2 = ResNet2d3d_two_stream(block, layers).stream2
        #self.head1 = nn.Linear(256 * 3 * 3 * 3, head1_dim)
        #self.head2 = nn.Linear(256 * 3 * 3 * 3, head2_dim)
        self.head1 = nn.Linear(256 * 2 * 4 * 4, head1_dim)
        self.head2 = nn.Linear(256 * 2 * 4 * 4, head2_dim)

        self.combined_head1 = nn.Linear(2 * 256 * 3 * 3 * 3, head1_dim)
        self.combined_head2 = nn.Linear(2 * 256 * 3 * 3 * 3, head2_dim)

        self.gaussian_blur = GaussianBlur(kernel_size=11, sigma=21.0)

    def forward(self, x, task='task1'):
        x_gray = x.mean(dim=1, keepdim=True) 
        x_gray = x_gray.expand(-1, 3, -1, -1, -1) 
        zero_frame = torch.zeros_like(x_gray[:, :, 0:1, :, :])

        x_temporal_derivative = x_gray[:, :, 1:, :, :] - x_gray[:, :, :-1, :, :]
        x_temporal_derivative = torch.cat((x_temporal_derivative, zero_frame), dim=2)
        blurred_temporal_derivative = torch.stack([
            self.gaussian_blur(frame) for frame in x_temporal_derivative.unbind(dim=2)
        ], dim=2)

        b, c, t, h, w = blurred_temporal_derivative.size()
        reshaped_input = blurred_temporal_derivative.view(b * t, c, h, w)

        downsampled_reshaped = F.interpolate(
        reshaped_input, scale_factor=0.5, mode='bilinear', align_corners=False)
        downsampled_temporal_derivative = downsampled_reshaped.view(b, c, t, downsampled_reshaped.size(2), downsampled_reshaped.size(3))
    
        upsampled_frames = []
        for i in range(t):
            frame = downsampled_temporal_derivative[:, :, i, :, :]
            upsampled_frame = F.interpolate(
                frame, 
                size=(h, w),  # Specify the original height and width
                mode='bilinear',  # Use bilinear interpolation for upsampling
                align_corners=False
            )
            upsampled_frames.append(upsampled_frame)
        
        upsampled_temporal_derivative = torch.stack(upsampled_frames, dim=2)

        """if torch.sum(x_temporal_derivative) == 0:
            print("Warning: Temporal derivative is all zeros.")
        else:
            print("Temporal derivative calculated successfully.")"""

        static_image = x[:, :, 0:1, :, :].clone()
        static_image_repeated = static_image.expand(-1, -1, x.size(2), -1, -1)

        #self.visualize(static_image_repeated, upsampled_temporal_derivative)

        out1 = self.stream1(static_image_repeated)
        out2 = self.stream2(upsampled_temporal_derivative)

        #print(out1.shape, out2.shape)
        return out1, out2
    
    def visualize(self, static_image, x_temporal_derivative, save_path='visualizations'):
        # Convert tensors to numpy arrays for visualization
        static_image_np = static_image[0, 0].cpu().numpy()
        temporal_derivative_np = x_temporal_derivative[0, 0].cpu().numpy()

        # Ensure the save directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save static image as a GIF
        static_images = []
        for i in range(temporal_derivative_np.shape[0]):  # Repeat for the same number of frames as the temporal derivative
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title(f"Static Image Frame {i+1}")
            ax.imshow(static_image_np[i], cmap='gray')
            ax.axis('off')

            # Convert the plot to an image and append to the list
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            static_images.append(image)
            plt.close(fig)

        # Save the static images as a GIF
        imageio.mimsave(os.path.join(save_path, 'static_image.gif'), static_images, fps=5)

        # Save temporal derivative as a GIF
        temporal_images = []
        for i in range(temporal_derivative_np.shape[0]):
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title(f"Temporal Derivative Frame {i+1}")
            ax.imshow(temporal_derivative_np[i], cmap='gray', vmin=0, vmax=1)  # Use normalized values
            ax.axis('off')

            # Convert the plot to an image and append to the list
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            temporal_images.append(image)
            plt.close(fig)

        # Save the temporal derivative images as a GIF
        imageio.mimsave(os.path.join(save_path, 'temporal_derivative.gif'), temporal_images, fps=5)

    def forward_combined(self, x):
        out1 = self.stream1(x)
        out2 = self.stream2(x)
        
        combined_out = torch.cat((out1, out2), dim=1)
        return combined_out

def train_backbone(model, train_dataset, val_dataset, device, stream, task, transform=None, num_epochs=100, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=8)
    model.train()
    model_to_use = model.module if isinstance(model, nn.DataParallel) else model
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(model_to_use.head1.parameters()) + list(model_to_use.head2.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for inputs, labels in train_loader:
            inputs = [apply_transforms(video, transform) for video in inputs]
            inputs = torch.stack(inputs).to(device)
            inputs = inputs.permute(0, 2, 1, 3, 4)
            labels = labels.to(device)

            optimizer.zero_grad()
            model_to_use = model.module if isinstance(model, nn.DataParallel) else model

            out1, out2 = model_to_use(inputs, task=task)

            if stream == 'stream1':
                # Ensure the reshaped size matches the input size of the linear layer
                out1_flat = out1.view(out1.size(0), -1)
                out = model_to_use.head1(out1_flat) if task == 'task1' else model_to_use.head2(out1_flat)
            elif stream == 'stream2':
                out2_flat = out2.view(out2.size(0), -1)
                out = model_to_use.head1(out2_flat) if task == 'task1' else model_to_use.head2(out2_flat)
            else:
                combined_out = model_to_use.forward_combined(inputs)
                combined_out_flat = combined_out.view(combined_out.size(0), -1)
                out = model_to_use.combined_head1(combined_out_flat) if task == 'task1' else model_to_use.combined_head2(combined_out_flat)

            loss = criterion(out, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = [apply_transforms(video, transform) for video in inputs]
                inputs = torch.stack(inputs).to(device)
                inputs = inputs.permute(0, 2, 1, 3, 4)
                labels = labels.to(device)

                out1, out2 = model_to_use(inputs, task=task)

                if stream == 'stream1':
                    # Ensure the reshaped size matches the input size of the linear layer
                    out1_flat = out1.view(out1.size(0), -1)
                    out = model_to_use.head1(out1_flat) if task == 'task1' else model_to_use.head2(out1_flat)
                elif stream == 'stream2':
                    out2_flat = out2.view(out2.size(0), -1)
                    out = model_to_use.head1(out2_flat) if task == 'task1' else model_to_use.head2(out2_flat)
                else:
                    combined_out = model_to_use.forward_combined(inputs)
                    combined_out_flat = combined_out.view(combined_out.size(0), -1)
                    out = model_to_use.combined_head1(combined_out_flat) if task == 'task1' else model_to_use.combined_head2(combined_out_flat)

                _, predicted = torch.max(out, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
        model.train()


def apply_transforms(video, transform, task='task1'):
    transformed_video = []
    #resize_transform = transforms.Resize((112, 112))
    #print(video.shape)

    selected_blocks = video[:5]

    for block in selected_blocks: #instead of video
        if task == 'task1':
            # Sample the entire block for task1 (e.g., MNIST digit detection)
            sampled_frames = block[:, 0, :, :].unsqueeze(1)
            #sampled_frames = block[0:5, 0, :, :].unsqueeze(1)
        else:
            sampled_frames = block[0:1, :, :, :].unsqueeze(1)
            #sampled_frames = block[0:1, 0:5, :, :].unsqueeze(1)
        for frame in sampled_frames:
            # Convert tensor to PIL image
            frame = transforms.ToPILImage()(frame)
            # Convert grayscale to RGB
            frame = frame.convert("RGB")

            original_width, original_height = frame.size
            padding_left = (112 - original_width) // 2
            padding_top = (112 - original_height) // 2
            padding_right = 112 - original_width - padding_left
            padding_bottom = 112 - original_height - padding_top

            frame = transforms.functional.pad(frame, (padding_left, padding_top, padding_right, padding_bottom), fill=0)            

            # Apply the transformation
            if transform:
                frame = transform(frame)
            else:
                # Ensure conversion to tensor if no transform is provided
                frame = transforms.ToTensor()(frame)

            frame = frame.permute(0, 2, 1)
            # Append the transformed tensor
            transformed_video.append(frame)

    if task == 'task1':
        return torch.stack(transformed_video[:10])
    else:
        return torch.stack(transformed_video)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the backbone model with different head dimensions
    model = Backbone([BasicBlock2d, BasicBlock2d, BasicBlock3d, BasicBlock3d], [1, 1, 1, 1], 100, 26)

    # Load the model checkpoint
    checkpoint_path = 'modelepoch.pth.tar'
    load_checkpoint(model, checkpoint_path, device)

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    digits_dataset = MovingMNISTDigits(root='', split='train')
    motions_dataset = MovingMNISTMotions(root='', split='train')

    val_digits_dataset = MovingMNISTDigits(root='', split='val')
    val_motions_dataset = MovingMNISTMotions(root='', split='val')

    # Train the combined stream on a task
    """print("Training Combined Stream on Task 1 (MovingMNIST Digits)...")
    train_backbone(model, digits_dataset, val_digits_dataset, stream='combined', task='task1', device=device)
    print("Training Combined Stream on Task 2 (MovingMNIST Motions)...")
    train_backbone(model, motions_dataset, val_motions_dataset, stream='combined', task='task2', device=device)"""

    # Train each stream on each task
    
    print("Training Stream 1 on Task 1 (MovingMNIST Digits)...")
    train_backbone(model, digits_dataset, val_digits_dataset, device=device, stream='stream1', task='task1')

    print("Training Stream 1 on Task 2 (MovingMNIST Motions)...")
    train_backbone(model, motions_dataset, val_motions_dataset, device=device, stream='stream1', task='task2')

    print("Training Stream 2 on Task 1 (MovingMNIST Digits)...")
    train_backbone(model, digits_dataset, val_digits_dataset, device=device, stream='stream2', task='task1')

    print("Training Stream 2 on Task 2 (MovingMNIST Motions)...")
    train_backbone(model, motions_dataset, val_motions_dataset, device=device, stream='stream2', task='task2')

if __name__ == '__main__':
    main()

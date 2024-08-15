import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pytorch_ssim

from model_vox2vox import UNet3DWithAttenuation, Discriminator
from DataLoading import SimulationDatasetAttenuation2_wo_norm


# Check if CUDA is available and set the appropriate tensor type
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Utility function to normalize an image
def normalize(img):
    img_min = img.min()
    img_max = img.max()
    return np.clip((img - img_min) / (img_max - img_min), 0, 1)

# Function to plot and save the results of the input, target, output, and difference images
def plot_results(input_img, target_img, output_img, epoch, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot the input image
    axes[0, 0].imshow(input_img[input_img.shape[0] // 2], cmap='gray')
    axes[0, 0].set_title('Input')
    axes[0, 0].axis('off')

    # Plot the target image
    axes[0, 1].imshow(target_img[target_img.shape[0] // 2], cmap='gray')
    axes[0, 1].set_title('Target')
    axes[0, 1].axis('off')

    # Plot the output image
    axes[1, 0].imshow(output_img[output_img.shape[0] // 2], cmap='gray')
    axes[1, 0].set_title('Output')
    axes[1, 0].axis('off')

    # Plot the normalized difference between target and output
    diff = normalize(target_img) - normalize(output_img)
    im = axes[1, 1].imshow(diff[diff.shape[0] // 2], cmap='seismic', vmin=-1, vmax=1)
    axes[1, 1].set_title('Normalized Difference')
    axes[1, 1].axis('off')

    # Add a colorbar for the difference plot
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'results_epoch_{epoch}.png'))
    plt.close()

# Function to save a model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

# Function to load a model checkpoint
def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
        return start_epoch, loss
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0, None

# Training function
def train(model, discriminator, train_loader, val_loader, criterion, optimizer_G, optimizer_D, num_epochs, device, save_dir, save_dir_img, d_threshold=0.8):
    gen_checkpoint_file = os.path.join(save_dir, 'gen_checkpoint.pth')
    disc_checkpoint_file = os.path.join(save_dir, 'disc_checkpoint.pth')
    start_epoch, _ = load_checkpoint(model, optimizer_G, gen_checkpoint_file)
    _, _ = load_checkpoint(discriminator, optimizer_D, disc_checkpoint_file)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        discriminator.train()
        train_loss = 0.0

        for inputs, targets, materials in train_loader:
            inputs, targets, materials = inputs.to(device), targets.to(device), materials.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()

            # Real images
            real_outputs = discriminator(targets, inputs, materials)
            valid = torch.ones_like(real_outputs, requires_grad=False)
            fake = torch.zeros_like(real_outputs, requires_grad=False)
            loss_real = torch.nn.MSELoss()(real_outputs, valid)

            # Fake images
            fake_images = model(inputs, materials)
            fake_outputs = discriminator(fake_images.detach(), inputs, materials)
            loss_fake = torch.nn.MSELoss()(fake_outputs, fake)

            d_loss = loss_real + loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            fake_outputs = discriminator(fake_images, inputs, materials)
            g_adv_loss = torch.nn.MSELoss()(fake_outputs, valid)

            g_l1_loss = criterion(fake_images, targets)
            g_loss = g_adv_loss + 25 * g_l1_loss
            g_loss.backward()
            optimizer_G.step()

            train_loss += g_loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, materials in val_loader:
                inputs, targets, materials = inputs.to(device), targets.to(device), materials.to(device)
                outputs = model(inputs, materials)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                # Save the first batch results for visualization
                if val_loss == loss.item() * inputs.size(0):
                    input_img = inputs[0, 0].cpu().numpy()
                    target_img = targets[0, 0].cpu().numpy()
                    output_img = outputs[0, 0].cpu().numpy()
                    plot_results(input_img, target_img, output_img, epoch, save_dir_img)
                break

        val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Save checkpoints
        save_checkpoint(model, optimizer_G, epoch + 1, val_loss, gen_checkpoint_file)
        save_checkpoint(discriminator, optimizer_D, epoch + 1, val_loss, disc_checkpoint_file)

        # Save model weights
        torch.save(model.state_dict(), os.path.join(save_dir, f'gen_model_epoch_{epoch+1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, f'disc_model_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    # Hyperparameters
    batch_size = 2
    learning_rate = 0.0005
    num_epochs = 2000

    # Directories for saving models and images
    save_dir = '/homes/ymellak/3DUnet3Gamma/Fast3GPet/UNET_Vox2Vox3_wo_norm_Att_weights/'
    save_dir_img = '/homes/ymellak/3DUnet3Gamma/Fast3GPet/UNET_Vox2Vox3_wo_norm_Att_images/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_img, exist_ok=True)

    # Load datasets and data loaders
    train_dataset = SimulationDatasetAttenuation2_wo_norm('/homes/ymellak/3DUnet3Gamma/DataSet', transform=True)
    val_dataset = SimulationDatasetAttenuation2_wo_norm('/homes/ymellak/3DUnet3Gamma/ValDataSet', transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize model, discriminator, loss function, and optimizers
    model = UNet3DWithAttenuation(att_num_classes=4, out_channels=1, final_sigmoid=False).to(device)
    discriminator = Discriminator(in_channels=6).to(device)
    criterion = nn.L1Loss()
    optimizer_G = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

    # Train the model
    train(model, discriminator, train_loader, val_loader, criterion, optimizer_G, optimizer_D, num_epochs, device, save_dir, save_dir_img)


import torch
import numpy as np
from model_vox2vox import UNet3DWithAttenuation
import os
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_bin_image(file_path, shape=(200, 200, 200)):
    # Load the .bin file
    with open(file_path, 'rb') as f:
        bin_data = np.fromfile(f, dtype=np.float32)
    
    # Reshape the data to 3D (assuming you know the dimensions)
    bin_image = bin_data.reshape(shape)

    # Add batch and channel dimensions and convert to torch tensor
    bin_image = torch.from_numpy(bin_image).unsqueeze(0).unsqueeze(0).float()
    return bin_image

def load_attenuation_map(file_path, shape=(100, 200, 200)):
    # Load the .bin file
    with open(file_path, 'rb') as f:
        bin_data = np.fromfile(f, dtype=np.int32)
    
    # Reshape the data to 3D (assuming you know the dimensions)
    att_map = bin_data.reshape(shape)
    
    att_map = np.pad(att_map, ((50, 50), (0, 0), (0, 0)), 'constant')
    att_map = np.clip(att_map, 0, 3)
    # Add batch and channel dimensions and convert to torch tensor
    att_map = torch.from_numpy(att_map).unsqueeze(0).unsqueeze(0).long()
    return att_map

def one_hot_encode(tensor, num_classes):
    return torch.nn.functional.one_hot(tensor.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

def predict_emission(model, bp_image, att_map, device):
    model.eval()
    with torch.no_grad():
        bp_image = bp_image.to(device)
        att_map = att_map.to(device)

        emission_pred = model(bp_image, att_map)
    return emission_pred.squeeze().cpu().numpy()

def save_bin_image(image, file_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the image as a .bin file
    with open(file_path, 'wb') as f:
        image.astype(np.float32).tofile(f)

def main(img_idx, epoch, input_bin_path, att_map_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = UNet3DWithAttenuation(att_num_classes=4, out_channels=1, final_sigmoid=False).to(device)
    # UNET_Att-SSIM_wo_norm_weights or UNET_Att_wo_norm_weights
    model_path = f'/homes/ymellak/Direct3G_f/3DReco/weights_gen_model_epoch_{epoch}.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f'Model is well loaded epoch={epoch}')
    
    # Load input .bin image
    bp_image = load_bin_image(input_bin_path)
    
    # Load attenuation map
    att_map = load_attenuation_map(att_map_path)
    
    # Predict emission image
    emission_pred = predict_emission(model, bp_image, att_map, device)
    

    #emission_pred = (emission_pred - np.min(emission_pred))  / (np.max(emission_pred) - np.min(emission_pred))
    # Save predicted emission image as .bin
    output_dir = '/homes/ymellak/Direct3G_f/3DReco/test_results/'
    os.makedirs(output_dir, exist_ok=True)
    output_bin_path = os.path.join(output_dir, f'Vox2VoxAtt_wo_norm_{img_idx}.bin')
    save_bin_image(emission_pred, output_bin_path)
    
    print(f"Prediction completed. Results saved to {output_bin_path}")

if __name__ == '__main__':
    img_idx = 999
    epoch   = 29 # Best one 29
    input_bin = '/homes/ymellak/Direct3G_f/3DReco/Images/Simu'+str(img_idx)+'/BPUImage_wA_'+str(img_idx)+'.bin'
    atten_bin = '/homes/ymellak/Direct3G_f/3DReco/Images/Simu'+str(img_idx)+'/Materials_image.bin'
    
    main(img_idx, epoch, input_bin, atten_bin)


# Three-gamma PET Image Reconstruction - from Histo-Image to reconstructed Image

## Files in this folder:

1. **DataLoading.py**
   - Contains classes for loading and preprocessing the simulation dataset.
   - Includes data augmentation techniques such as flips, rotations, and translations.
   - Defines `SimulationDataset`, `NormalizedSimulationDataset`, and `SimulationDatasetAttenuation` classes.

2. **model_vox2vox.py**
   - Implements the UNet3D architecture.
   

3. **Test_wAtt_Vox2Vox_no_norm3.py**
   - Script for testing the trained model on new data.
   - Loads a trained model and predicts emission images from input PET images and attenuation maps.
   - Saves the predicted images as binary files.

4. **Train_Direct3g_example_wAtt.py**
   - Main training script for the UNet3D model with attenuation correction.
   - Implements the training loop, including adversarial training with a discriminator.
   - Includes functions for saving checkpoints, visualizing results, and logging training progress.

## Usage

1. Prepare your dataset in the format expected by the `SimulationDatasetAttenuation2_wo_norm` class.
2. Run the `Train_Direct3g_example_wAtt.py` script to train the model.
3. Use the `Test_wAtt_Vox2Vox_no_norm3.py` script to test the trained model on new data.

Note: Make sure to adjust the file paths and hyperparameters in the scripts according to your setup and requirements.
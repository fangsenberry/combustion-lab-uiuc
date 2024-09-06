import pickle
import matplotlib.pyplot as plt
import os
#create folder loss_info
base_path = r'D:\test_cases\UPF_A01_C_DP_35_trial_33'
folder_path = os.path.join(base_path, 'loss_info')
os.makedirs(folder_path, exist_ok=True)

with open(r'D:\test_cases\UPF_A01_C_DP_35_trial_33\loss_data.pkl', 'rb') as f:
    loaded_loss_data = pickle.load(f)

smooth_losses = loaded_loss_data['smooth_loss']
photo_losses = loaded_loss_data['photo_loss']
census_losses = loaded_loss_data['census_loss']
msd_losses = loaded_loss_data['msd_loss']
print(f'Smooth Losses: {smooth_losses[:5]}')  # Print the first 5 values as a check
print(f'Photo Losses: {photo_losses[:5]}')
print(f'Census Losses: {census_losses[:5]}')
print(f'MSD Losses: {msd_losses[:5]}')
total_losses = [smooth + photo + census + msd for smooth, photo, census, msd in zip(smooth_losses, photo_losses, census_losses, msd_losses)]

def plot_losses(losses, title, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label=ylabel)
    plt.xlabel('Batch')
    plt.ylabel(ylabel)
    #plot y axis on log
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    #make a folder to save the plots
    plt.savefig(os.path.join(folder_path, f'{ylabel}.png'))
    # plt.show()

# Plot each loss component
plot_losses(smooth_losses, 'Smooth Loss Over Time', 'Smooth Loss')
plot_losses(photo_losses, 'Photo Loss Over Time', 'Photo Loss')
plot_losses(census_losses, 'Census Loss Over Time', 'Census Loss')
plot_losses(msd_losses, 'Multi-Scale Distillation (MSD) Loss Over Time', 'MSD Loss')
plot_losses(total_losses, 'Total Loss Over Time', 'Total Loss')

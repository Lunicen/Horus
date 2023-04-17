import os
from pathlib import Path
import numpy as np
import scipy.stats as stats
import glob
import seaborn as sns
import matplotlib.pyplot as plt

# Set the path to the directory containing the 15 folders
base_path = Path('data/numpy')

# Create a list of folder names
folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
species_names = [os.path.basename(f) for f in folders]

# Initialize lists to store the data
spectrogram_data = []
species_labels = []

# Load the data from .npy files
for idx, folder in enumerate(folders):
    file_list = glob.glob(os.path.join(folder, '*.npy'))
    
    for file in file_list:
        spectrogram = np.load(file)
        spectrogram_data.append(spectrogram)
        species_labels.append(idx)

# Calculate average spectrograms for each species
avg_spectrograms = []

for idx in range(len(folders)):
    species_spectrograms = [spec for spec, label in zip(spectrogram_data, species_labels) if label == idx]
    avg_spectrogram = np.mean(species_spectrograms, axis=0)
    avg_spectrograms.append(avg_spectrogram)

# Perform statistical analysis
correlation_matrix = np.zeros((len(folders), len(folders)))

for i in range(len(folders)):
    for j in range(len(folders)):
        # Calculate the Pearson correlation coefficient between the average spectrograms
        corr, _ = stats.pearsonr(avg_spectrograms[i].flatten(), avg_spectrograms[j].flatten())
        correlation_matrix[i, j] = corr

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=species_names, yticklabels=species_names)
plt.title('Correlation Matrix of Average Spectrograms')
plt.xlabel('Species')
plt.ylabel('Species')
plt.show()

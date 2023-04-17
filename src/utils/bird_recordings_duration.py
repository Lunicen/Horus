import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Mapping Latin bird names to Polish bird names
polish_names = {
    "alauda_arvensis": "Skowronek",
    "erithacus_rubecula": "Rudzik",
    "parus_major": "Bogatka",
    "troglodytes_troglodytes": "Strzyżyk",
    "turdus_merula": "Kos",
}

# List of folders containing bird recordings
folders = [
    "alauda_arvensis",
    "erithacus_rubecula",
    "other",
    "parus_major",
    "troglodytes_troglodytes",
    "turdus_merula",
]

# Initialize dictionaries to store total durations, average lengths, and standard deviations for each bird species
durations = {}
average_lengths = {}
standard_deviations = {}


def process_folder(folder_path):
    lengths = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".mp3") or file.endswith(".wav"):
            audio = AudioSegment.from_file(file_path)
            lengths.append(audio.duration_seconds)
        elif os.path.isdir(file_path) and folder_path.endswith("other"):
            lengths += process_folder(file_path)
    return lengths


print("Processing bird recordings...")

for folder in folders:
    print(f"Processing folder '{folder}'...")
    lengths = process_folder(folder)

    # Calculate statistics
    total_duration = sum(lengths)
    avg_length = np.mean(lengths)
    std_deviation = np.std(lengths)

    # Convert the total duration to hours
    total_duration /= 3600
    polish_name = polish_names.get(folder, "Inne")
    durations[polish_name] = total_duration
    average_lengths[polish_name] = avg_length
    standard_deviations[polish_name] = std_deviation

    print(f"Total duration for {polish_name}: {total_duration:.2f} hours")
    print(f"Average length for {polish_name}: {avg_length:.2f} seconds")
    print(f"Standard deviation for {polish_name}: {std_deviation:.2f} seconds")

print("Finished processing bird recordings.")

# Plot the results
print("Creating plot...")

fig, ax1 = plt.subplots()

ax1.bar(durations.keys(), durations.values())
ax1.set_xlabel("Gatunek ptaka")
ax1.set_ylabel("Czas nagrania (godziny)")
ax1.set_title("Czas nagrania dla każdego gatunku ptaka")

ax2 = ax1.twinx()
ax2.plot(
    list(durations.keys()),
    list(average_lengths.values()),
    color="r",
    marker="o",
    label="Średnia długość",
)
ax2.plot(
    list(durations.keys()),
    list(standard_deviations.values()),
    color="g",
    marker="o",
    label="Odchylenie standardowe",
)
ax2.set_ylabel("Czas nagrania (sekundy)")

fig.legend(loc="upper right")
plt.show()

print("Plot displayed.")

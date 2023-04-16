import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm


class AudioPreprocessor:
    def __init__(
        self,
        source_directory="data/raw",
        target_directory="data/processed",
        spectrogram_directory="data/spectrograms",
        sample_rate=48000,
        fixed_length_seconds=5,
        normalize_volume=True,
    ):
        self.source_directory = source_directory
        self.target_directory = target_directory
        self.spectrogram_directory = spectrogram_directory
        self.sample_rate = sample_rate
        self.fixed_length_seconds = fixed_length_seconds
        self.normalize_volume = normalize_volume

        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

    def normalize_file_path(self, path):
        return path.replace("\\", "/")

    def preprocess_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate)

        if len(audio) == 0:
            return None

        if self.normalize_volume:
            audio = librosa.util.normalize(audio)

        audio_length = len(audio)
        fixed_length_samples = self.fixed_length_seconds * sr

        if audio_length < fixed_length_samples:
            # maybe add 0s if it's not long enough?
            # audio = np.pad(audio, (0, fixed_length_samples - audio_length))
            return None
        elif audio_length > fixed_length_samples:
            audio = audio[:fixed_length_samples]
        
        bird_class = file_path.split("/")[-2]
        target_file_path = os.path.join(
            self.target_directory, bird_class, os.path.basename(file_path)
        )

        # librosa.output.write_wav(target_file_path, audio, sr)
        sf.write(target_file_path, audio, sr)
        return self.normalize_file_path(target_file_path)

    def create_spectrogram(self, file_path, n_fft=2048, hop_length=512):
        y, sr = librosa.load(file_path, sr=None, mono=True)
        spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_db

    def create_mel_spectrogram(self, file_path, n_mels=128, hop_length=512):
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        stft = librosa.stft(y)
        mel_spectrogram = librosa.feature.melspectrogram(
            S=np.abs(stft), sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        db_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
        return db_mel_spectrogram

    def save_mel_spectrogram(self, mel_spectrogram_db, target_path, sr, hop_length):
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(
            mel_spectrogram_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            hop_length=hop_length,
        )
        plt.colorbar()
        plt.title(target_path.split("/")[-1])
        plt.tight_layout()

        plot_path = os.path.join(
            self.spectrogram_directory,
            target_path.split("/")[-2],
            target_path.split("/")[-1],
        )
        plot_path = plot_path.replace(".mp3", ".png")
        plt.savefig(plot_path)
        plt.close()

    def plot_mel_spectrogram(self, mel_spectrogram_db, target_path, sr, hop_length):
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(
            mel_spectrogram_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            hop_length=hop_length,
        )
        plt.colorbar()
        plt.title(target_path.split("/")[-1])
        plt.tight_layout()
        plt.show()
        plt.close()

    def process_directory(self):
        for root_dir, directories, _ in os.walk(self.source_directory):
            for directory in directories:
                dirpath = self.normalize_file_path(os.path.join(root_dir, directory))
                # check if required directories exist, if not then create them
                os.makedirs(
                    os.path.join(self.spectrogram_directory, directory), exist_ok=True
                )
                os.makedirs(
                    os.path.join(self.target_directory, directory), exist_ok=True
                )
                for file in tqdm(os.listdir(dirpath), unit="File"):
                    try:
                        file_path = self.normalize_file_path(os.path.join(dirpath, file))
                        target_file_path = self.preprocess_audio(file_path)
                        if target_file_path is not None:
                            ms = self.create_mel_spectrogram(target_file_path)
                            self.save_mel_spectrogram(
                                ms, target_file_path, self.sample_rate, hop_length=512
                            )
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    audio_preprocessor = AudioPreprocessor()
    audio_preprocessor.process_directory()

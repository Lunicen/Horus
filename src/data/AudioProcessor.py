import os
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import noisereduce as nr


class AudioPreprocessor:
    def __init__(
        self,
        source_directory="data/raw",
        target_directory="data/processed",
        spectrogram_directory="data/spectrograms",
        numpy_dir="data/numpy",
        sample_rate=48000,
        fixed_length_seconds=5,
        normalize_volume=True,
        data_augmentation=True,
    ):
        self.source_directory = Path(source_directory)
        self.target_directory = Path(target_directory)
        self.spectrogram_directory = Path(spectrogram_directory)
        self.numpy_dir = Path(numpy_dir)
        self.sample_rate = sample_rate
        self.fixed_length_seconds = fixed_length_seconds
        self.normalize_volume = normalize_volume
        self.target_directory.mkdir(parents=True, exist_ok=True)
        self.data_augmentation = data_augmentation

    def standardize(self, spectrogram):
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        return (spectrogram - mean) / std

    def shift_pitch(self, audio, sr, pitch_factor=3):
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

    def change_speed(self, audio, sr, speed_factor=1.2):
        return librosa.effects.time_stretch(audio, rate=speed_factor)

    def preprocess_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sample_rate)

        if len(audio) == 0:
            return None

        audio = nr.reduce_noise(y=audio, sr=sr)
        if self.normalize_volume:
            audio = librosa.util.normalize(audio)

        audio_length = len(audio)
        fixed_length_samples = self.fixed_length_seconds * sr

        if audio_length < fixed_length_samples:
            return None
        elif audio_length > fixed_length_samples:
            audio = audio[:fixed_length_samples]

        bird_class = file_path.parent.name
        target_file_path = self.target_directory / bird_class / file_path.name

        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target_file_path, audio, sr)
        return target_file_path

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
        db_mel_spectrogram = self.standardize(db_mel_spectrogram)
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
        plt.title(target_path.name)
        plt.tight_layout()

        plot_path = (
            self.spectrogram_directory
            / target_path.parent.name
            / target_path.with_suffix(".png").name
        )
        mel_spec_path = (
            self.numpy_dir
            / target_path.parent.name
            / target_path.with_suffix(".npy").name )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
        mel_spec_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mel_spec_path, mel_spectrogram_db)

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
        plt.title(target_path.name)
        plt.tight_layout()
        plt.show()
        plt.close()

    def process_directory(self):
        for root_dir, directories, _ in os.walk(self.source_directory):
            for directory in directories:
                self.process_folder(Path(root_dir) / directory)

    def augment_data(self, target_file_path):
        target_file_path = Path(target_file_path)
        target_file_path_change_speed = target_file_path.stem + "_speed"
        target_file_path_shift_pitch = target_file_path.stem + "_pitch"
        target_file_path_change_speed = target_file_path.with_stem(
            target_file_path_change_speed
        )
        target_file_path_shift_pitch = target_file_path.with_stem(
            target_file_path_shift_pitch
        )

        # Load the original audio
        y, sr = librosa.load(target_file_path, sr=self.sample_rate)

        # Change the speed and pitch of the audio and save the new files
        y_change_speed = self.change_speed(y, sr)
        y_shift_pitch = self.shift_pitch(y, sr)
        sf.write(target_file_path_change_speed, y_change_speed, sr)
        sf.write(target_file_path_shift_pitch, y_shift_pitch, sr)

        ms_change_speed = self.create_mel_spectrogram(target_file_path_change_speed)
        ms_shift_pitch = self.create_mel_spectrogram(target_file_path_shift_pitch)
        self.save_mel_spectrogram(
            ms_change_speed,
            target_file_path_change_speed,
            self.sample_rate,
            hop_length=512,
        )
        self.save_mel_spectrogram(
            ms_shift_pitch,
            target_file_path_shift_pitch,
            self.sample_rate,
            hop_length=512,
        )

    def process_folder(self, folder_path: Path):
        dirpath = folder_path
        self.spectrogram_directory.mkdir(parents=True, exist_ok=True)
        self.target_directory.mkdir(parents=True, exist_ok=True)
        for file in os.listdir(dirpath):
            file_path = dirpath / file
            if os.path.isdir(file_path):
                self.process_folder(file_path)
            else:
                try:
                    target_file_path = self.preprocess_audio(file_path)

                    if target_file_path is not None:
                        ms = self.create_mel_spectrogram(target_file_path)

                        self.save_mel_spectrogram(
                            ms, target_file_path, self.sample_rate, hop_length=512
                        )
                        if self.data_augmentation:
                            self.augment_data(target_file_path)

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")


if __name__ == "__main__":
    audio_preprocessor = AudioPreprocessor()
    audio_preprocessor.process_directory()

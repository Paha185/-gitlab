import os
import torchaudio
import numpy as np
import torch


class VCTKDataset:
    def __init__(self, root_dir):
        self.data = []
        self.speakers = os.listdir(root_dir)
        max_length = 0
        for speaker in self.speakers:
            speaker_dir = os.path.join(root_dir, speaker)
            for file_name in os.listdir(speaker_dir):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(speaker_dir, file_name)
                    waveform, sample_rate = torchaudio.load(file_path, format='wav')
                    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=32)(waveform)
                    self.data.append((mel_spectrogram[0][0].numpy(), speaker))
                    # Обновляем максимальную длину спектрограммы
                    if len(mel_spectrogram[0][0]) > max_length:
                        max_length = len(mel_spectrogram[0][0])

        print(max_length)
        # Приводим все спектрограммы к одинаковой длине
        for i in range(len(self.data)):
            mel_spectrogram, speaker = self.data[i]
            if len(mel_spectrogram) < max_length:
                pad_width = (0, max_length - len(mel_spectrogram))
                mel_spectrogram = np.pad(mel_spectrogram, pad_width, mode='constant')
            self.data[i] = (mel_spectrogram, speaker)

    def save_data(self, spectrogram_file_path, speakers_file_path):
        # Преобразуем данные в numpy массивы
        mel_spectrograms = np.array([item[0] for item in self.data])
        speakers = np.array([item[1] for item in self.data])

        # Сохраняем данные в файлы
        np.save(spectrogram_file_path, mel_spectrograms)
        np.save(speakers_file_path, speakers)
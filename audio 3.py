import torch
import torch.optim as optim

from dataset import VCTKDataset
from model import RNNModel, triplet_loss
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from torch.nn.functional import cosine_similarity
from torch.utils.tensorboard import SummaryWriter
def load_data(spectrogram_file_path, speakers_file_path):
    mel_spectrograms = np.load(spectrogram_file_path)
    speakers = np.load(speakers_file_path)
    return mel_spectrograms, speakers


def create_triplets(mel_spectrograms, speakers):
    triplets = []
    for i in range(len(mel_spectrograms)):
        anchor = mel_spectrograms[i]
        anchor_speaker = speakers[i]

        # Выбираем положительный пример
        positive_indices = np.where(speakers == anchor_speaker)[0]
        positive_index = np.random.choice(positive_indices)
        positive = mel_spectrograms[positive_index]

        # Выбираем отрицательный пример
        negative_indices = np.where(speakers != anchor_speaker)[0]
        negative_index = np.random.choice(negative_indices)
        negative = mel_spectrograms[negative_index]

        triplets.append([anchor, positive, negative])

    return np.array(triplets)

def normalize_data(mel_spectrograms):
    mean = np.mean(mel_spectrograms, axis=0)
    std = np.std(mel_spectrograms, axis=0)
    normalized_data = (mel_spectrograms - mean) / std
    return normalized_data
dir_path = 'Z:/pythonProject/dataset/VCTK-Corpus/wav48'


# Пример использования
mel_spectrograms, speakers = load_data('mel_spectrogram.npy', 'speakers.npy')
mel_spectrograms = normalize_data(mel_spectrograms)
triplets = create_triplets(mel_spectrograms, speakers)

# Преобразование данных в тензоры PyTorch
triplets_tensor = torch.tensor(triplets, dtype=torch.float32)
anchor_data = triplets_tensor[:, 0]
positive_data = triplets_tensor[:, 1]
negative_data = triplets_tensor[:, 2]

# Создание датасета и даталоадера
dataset = TensorDataset(anchor_data, positive_data, negative_data)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Создание модели
input_size = 4628
hidden_size = 64
num_layers = 3
output_size = 64
model = RNNModel(input_size, hidden_size, num_layers, output_size)

# Определение оптимизатора
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Инициализация SummaryWriter для TensorBoard
writer = SummaryWriter('runs/audio_triplet_loss5')

# Обучение модели
train_losses = []
test_losses = []
test_accuracies = []
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for anchor, positive, negative in train_dataloader:
        optimizer.zero_grad()

        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        loss = triplet_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)
    writer.add_scalar('Loss/train', train_loss, epoch)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for anchor, positive, negative in test_dataloader:
            anchor_output = model(anchor)
            positive_output = model(positive)
            negative_output = model(negative)

            loss = triplet_loss(anchor_output, positive_output, negative_output)
            test_loss += loss.item()

            pos_dist = 1 - cosine_similarity(anchor_output, positive_output, dim=1)
            neg_dist = 1 - cosine_similarity(anchor_output, negative_output, dim=1)
            correct += torch.sum(pos_dist < neg_dist).item()
            total += anchor_output.size(0)

    test_loss /= len(test_dataloader)
    test_losses.append(test_loss)
    writer.add_scalar('Loss/test', test_loss, epoch)

    test_accuracy = correct / total
    test_accuracies.append(test_accuracy)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)

    print(
        f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
# Закрытие SummaryWriter
writer.close()
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Добавляем размерность батча, если входные данные небатчевые
        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = 1 - cosine_similarity(anchor, positive, dim=1)
    neg_dist = 1 - cosine_similarity(anchor, negative, dim=1)
    loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))
    return loss
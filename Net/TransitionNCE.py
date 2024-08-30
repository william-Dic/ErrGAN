import torch
import torch.nn as nn
import torch.optim as optim
from BearRobot.Net.my_model.t5 import T5Encoder
import torch.nn.functional as F

class Transition_NCE(nn.Module):
    def __init__(self, state_dim=9, time_dim=1, hidden_dim=128, num_layers=2, T5_out_dim=768):
        super(Transition_NCE, self).__init__()
        self.lang_encoder = T5Encoder()
        
        input_dim = state_dim
        input_size_2 = time_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.lstm_time = nn.LSTM(input_size=input_size_2, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, T5_out_dim)

    def forward(self, state_seq, time_seq, lang):       
        lstm_out, _ = self.lstm(state_seq)
        lstm_out_time, _ = self.lstm_time(time_seq)
        lstm_out_all = torch.cat((lstm_out, lstm_out_time), dim=-1)
        
        state_seq_e = self.fc(lstm_out_all)
        
        text_e = self.lang_encoder.embed_text(lang)
        return state_seq_e, text_e

class Transition_NCE_loss(nn.Module):
    def __init__(self):
        super(Transition_NCE_loss, self).__init__()
    
    def Transition_Reward_diff(self, state_seq_e, text_e, logit_scale=100.):
        state_seq_e = state_seq_e.transpose(0, 1)
        diff = state_seq_e[1:] - state_seq_e[:-1]
        diff = diff / (diff.norm(dim=-1, keepdim=True) + 1e-6)
        text_e = text_e / text_e.norm(dim=-1, keepdim=True)
        frame_logits = diff @ text_e.transpose(0, 1) * logit_scale
        reward_matrix = torch.sum(frame_logits, dim=0)
        return reward_matrix
    
    def Transition_Reward(self, state_seq_e, text_e, logit_scale=100.):
        state_seq_e = state_seq_e.transpose(0, 1)
        state_seq_e = state_seq_e / state_seq_e.norm(dim=-1, keepdim=True)
        
        text_e = text_e / text_e.norm(dim=-1, keepdim=True)
        frame_logits = state_seq_e @ text_e.transpose(0, 1) * logit_scale
        reward_matrix = torch.sum(frame_logits, dim=0)
        return reward_matrix

    def forward(self, state_seq_e, text_e):
        reward_matrix = self.Transition_Reward_diff(state_seq_e, text_e)
        labels = torch.arange(reward_matrix.shape[0], device=reward_matrix.device).long()
        return (F.cross_entropy(reward_matrix, labels) + F.cross_entropy(reward_matrix.transpose(0, 1), labels)) / 2

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transition_NCE(state_dim=9, time_dim=5).to(device)
    loss_fn = Transition_NCE_loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(10000):
        state_seq = torch.randn(32, 10, 9).to(device)
        time_seq = torch.randn(32, 1, 10).to(device)
        print(time_seq)
        lang = ["this is a sample input"] * 32
        
        optimizer.zero_grad()
        state_seq_e, text_e = model(state_seq, time_seq, lang)
        loss = loss_fn(state_seq_e, text_e)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import random
from BearRobot.Net.my_model.t5 import T5Encoder

class Discriminator(nn.Module):
    def __init__(self, state_dim=9, action_dim=7, hidden_dim=128, num_lstm_layers=2, t5_dim=768):
        super(Discriminator, self).__init__()
        
        resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.image_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        image_feature_dim = resnet.fc.in_features
        
        self.lang_encoder = T5Encoder()
        self.t5_dim = t5_dim
        
        self.lstm = nn.LSTM(input_size=state_dim + image_feature_dim + t5_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        
        self.action_fc = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, image_seq, state_seq, action, lang):
        batch_size, seq_len, _, _, _ = image_seq.size()
        
        # Extract image features for each timestep
        image_features = []
        for t in range(seq_len):
            img_feat = self.image_feature_extractor(image_seq[:, t, :, :, :]).view(batch_size, -1)
            image_features.append(img_feat)
        image_features = torch.stack(image_features, dim=1)  # [B, Seq_len, image_feature_dim]
        
        # Encode language
        language_features = self.lang_encoder.embed_text(lang)  # [B, t5_dim]
        language_features = language_features.unsqueeze(1).repeat(1, seq_len, 1)  # [B, Seq_len, t5_dim]
        
        # Concatenate image features with state features and language features
        combined_input = torch.cat((state_seq, image_features, language_features), dim=-1)  # [B, Seq_len, state_dim + image_feature_dim + t5_dim]
        
        # Pass the combined input through the LSTM
        lstm_features, _ = self.lstm(combined_input)  # [B, Seq_len, hidden_dim]
        
        # Combine LSTM output with the action
        action_input = torch.cat((lstm_features[:, -1, :], action), dim=-1)  # [B, hidden_dim + action_dim]
        
        # Compute logits
        logits = self.action_fc(action_input)  # [B, 1]
        classes = torch.sigmoid(logits)  # [B, 1]
        
        return logits, classes


class Generator(nn.Module):
    def __init__(self, state_dim=9, action_dim=7, hidden_dim=128, image_dim=128, lstm_hidden_dim=64, num_lstm_layers=1, t5_dim=768):
        super(Generator, self).__init__()
        
        self.lang_encoder = T5Encoder()
        self.t5_dim = t5_dim
        
        self.image_fc = nn.Sequential(
            nn.Linear(image_dim * image_dim * 3 + t5_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim + t5_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.action_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.image_dim = image_dim
        
        # LSTM layers for state and image sequence
        self.lstm = nn.LSTM(input_size=hidden_dim * 2, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
    
    def forward(self, state_seq, image_seq, lang):
        batch_size, seq_len, _, _, _ = image_seq.size()
        
        # Encode language
        language_features = self.lang_encoder.embed_text(lang)  # [B, t5_dim]
        
        combined_features = []
        
        for t in range(seq_len):
            # Flatten images for FC processing
            image_feat = self.image_fc(torch.cat((image_seq[:, t, :].view(batch_size, -1), language_features), dim=-1))
            state_feat = self.state_fc(torch.cat((state_seq[:, t, :], language_features), dim=-1))
            combined_feat = torch.cat((state_feat, image_feat), dim=-1)
            combined_features.append(combined_feat)
        
        combined_features = torch.stack(combined_features, dim=1)  # [B, Seq_len, hidden_dim * 2]
        
        # Apply LSTM
        lstm_output, _ = self.lstm(combined_features)  # [B, Seq_len, lstm_hidden_dim]
        
        # Predict the action at the last timestep
        action_pred = self.action_fc(lstm_output[:, -1, :])  # [B, action_dim]
        
        return action_pred

def train_gan(generator, discriminator, num_epochs, device):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    for epoch in range(num_epochs):
        seq_len = random.randint(2, 5)  # Simulate a constant sequence length for the batch
        batch_size = 2
        state_dim = 9
        action_dim = 7
        image_dim = 128
        
        # Prepare sequences with constant length
        state_seqs = torch.randn(batch_size, seq_len, state_dim).to(device)  # [B, Seq_len, state_dim]
        real_images = torch.randn(batch_size, seq_len, 3, image_dim, image_dim).to(device)  # [B, Seq_len, 3, image_dim, image_dim]
        real_actions = torch.randn(batch_size, action_dim).to(device)  # [B, action_dim]
        lang = ["this is a sample input"] * batch_size  # [B]
        
        real_labels = torch.ones(batch_size, 1).to(device)  # [B, 1]
        fake_labels = torch.zeros(batch_size, 1).to(device)  # [B, 1]
        
        optimizer_D.zero_grad()
        
        # Discriminator on real data
        real_logits, real_classes = discriminator(real_images, state_seqs, real_actions, lang)
        real_loss = criterion(real_classes, real_labels)
        
        # Generate fake actions using the generator
        fake_actions = generator(state_seqs, real_images, lang)
        
        # Discriminator on fake data
        fake_logits, fake_classes = discriminator(real_images, state_seqs, fake_actions, lang)
        fake_loss = criterion(fake_classes, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        
        # Train the generator to fool the discriminator
        fake_actions = generator(state_seqs, real_images, lang)
        fake_logits, fake_classes = discriminator(real_images, state_seqs, fake_actions, lang)
        g_loss = criterion(fake_classes, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize generator and discriminator
generator = Generator(state_dim=9, action_dim=7, hidden_dim=128, image_dim=128, lstm_hidden_dim=64, num_lstm_layers=1, t5_dim=768).to(device)
discriminator = Discriminator(state_dim=9, action_dim=7, hidden_dim=128, num_lstm_layers=2, t5_dim=768).to(device)

# Uncomment to start training
# train_gan(generator, discriminator, num_epochs=100, device=device)

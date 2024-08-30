import torch
import torch.nn as nn
import torch.optim as optim
from BearRobot.Net.my_model.t5 import T5Encoder
import torchvision.models as models
import random

class Discriminator(nn.Module):
    def __init__(self, state_dim=9, hidden_dim=128, num_lstm_layers=2, t5_dim=768, num_classes=1):
        super(Discriminator, self).__init__()
        
        resnet = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.image_feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        image_feature_dim = resnet.fc.in_features
        
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        
        self.lang_encoder = T5Encoder()
        self.t5_dim = t5_dim
        
        self.logit_fc = nn.Sequential(
            nn.Linear(image_feature_dim + hidden_dim + t5_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.class_fc = nn.Sigmoid()
    
    def forward(self, image_seq, state_seq, lang):
        batch_size, seq_len, _, _, _ = image_seq.size()
        
        # Extract image features for each timestep
        image_features = []
        for t in range(seq_len):
            img_feat = self.image_feature_extractor(image_seq[:, t, :, :, :]).view(batch_size, -1)
            image_features.append(img_feat)
        image_features = torch.stack(image_features, dim=1)  # [B, Seq_len, image_feature_dim]
        
        # Pass the state sequence through the LSTM
        state_features, _ = self.lstm(state_seq)  # [B, Seq_len, hidden_dim]
        
        # Encode language
        language_features = self.lang_encoder.embed_text(lang)  # [B, t5_dim]
        language_features = language_features.unsqueeze(1).repeat(1, seq_len, 1)  # [B, Seq_len, t5_dim]
        
        # Combine features
        combined_features = torch.cat((image_features, state_features, language_features), dim=-1)  # [B, Seq_len, image_feature_dim + hidden_dim + t5_dim]
        
        # Compute logits
        logits = self.logit_fc(combined_features.mean(dim=1))  # [B, hidden_dim] -> [B, num_classes]
        classes = self.class_fc(logits)  # [B, num_classes]
        
        return logits, classes

class Generator(nn.Module):
    def __init__(self, state_dim=9, hidden_dim=128, t5_dim=768, image_dim=128, lstm_hidden_dim=64, num_lstm_layers=1):
        super(Generator, self).__init__()
        
        self.lang_encoder = T5Encoder()
        self.t5_dim = t5_dim
        
        self.image_fc = nn.Sequential(
            nn.Linear(t5_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim * image_dim * 3),
            nn.Tanh()
        )
        
        self.state_fc = nn.Sequential(
            nn.Linear(t5_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        
        self.image_dim = image_dim
        
        # Inverse LSTM layers
        self.inverse_lstm_image = nn.LSTM(input_size=image_dim * image_dim * 3, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        self.inverse_lstm_state = nn.LSTM(input_size=state_dim, hidden_size=lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
        
        # Final FC layers after inverse LSTM
        self.final_image_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 3 * image_dim * image_dim),
            nn.Tanh()
        )
        
        self.final_state_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, state_dim)
        )
    
    def forward(self, lang, seq_len):
        batch_size = len(lang)
        
        # [B, t5_dim]
        language_features = self.lang_encoder.embed_text(lang)  # [B, t5_dim]
        
        image_seq = []
        generated_state_seq = []
        
        for t in range(seq_len):
            # [B, 3, image_dim, image_dim]
            image = self.image_fc(language_features)
            image = image.view(batch_size, 3, self.image_dim, self.image_dim)
            image_seq.append(image)
            
            # [B, state_dim]
            generated_state = self.state_fc(language_features)
            generated_state_seq.append(generated_state)
        
        # Stack sequences and apply inverse LSTM
        image_seq = torch.stack(image_seq, dim=1)  # [B, Seq_len, 3, image_dim, image_dim]
        generated_state_seq = torch.stack(generated_state_seq, dim=1)  # [B, Seq_len, state_dim]
        
        # Flatten images for LSTM processing
        image_seq_flat = image_seq.view(batch_size, seq_len, -1)  # [B, Seq_len, 3 * image_dim * image_dim]
        
        # Apply inverse LSTM
        inverse_image_seq, _ = self.inverse_lstm_image(image_seq_flat)  # [B, Seq_len, lstm_hidden_dim]
        inverse_state_seq, _ = self.inverse_lstm_state(generated_state_seq)  # [B, Seq_len, lstm_hidden_dim]
        
        # Pass through final FC layers
        final_image_seq = self.final_image_fc(inverse_image_seq).view(batch_size, seq_len, 3, self.image_dim, self.image_dim)  # [B, Seq_len, 3, image_dim, image_dim]
        final_state_seq = self.final_state_fc(inverse_state_seq)  # [B, Seq_len, state_dim]
        
        return final_state_seq, final_image_seq

def train_gan(generator, discriminator, num_epochs, device):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
    
    for epoch in range(num_epochs):
        seq_len = random.randint(2, 5)  # Simulate a constant sequence length for the batch
        batch_size = 2
        state_dim = 9
        image_dim = 128
        
        # Prepare sequences with constant length
        state_seqs = torch.randn(batch_size, seq_len, state_dim).to(device)  # [B, Seq_len, state_dim]
        real_images = torch.randn(batch_size, seq_len, 3, image_dim, image_dim).to(device)  # [B, Seq_len, 3, image_dim, image_dim]
        
        lang = ["this is a sample input"] * batch_size  # [B]
        
        real_labels = torch.ones(batch_size, 1).to(device)  # [B, 1]
        fake_labels = torch.zeros(batch_size, 1).to(device)  # [B, 1]
        
        optimizer_D.zero_grad()
        
        # [B, num_classes], [B, 1]
        logits, labels = discriminator(real_images, state_seqs, lang)  
        real_loss = criterion(labels, real_labels)
        
        fake_state_seq, fake_images = generator(lang, seq_len)

        fake_logits, fake_classes = discriminator(fake_images, fake_state_seq, lang)
        fake_loss = criterion(fake_classes, fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward(retain_graph=True)
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        
        # [B, num_classes], [B, 1]
        fake_logits, fake_classes = discriminator(fake_images, fake_state_seq, lang)
        g_loss = criterion(fake_classes, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(state_dim=9, hidden_dim=128, t5_dim=768, image_dim=128, lstm_hidden_dim=64, num_lstm_layers=1).to(device)
discriminator = Discriminator(state_dim=9, hidden_dim=128, num_lstm_layers=2, t5_dim=768).to(device)

# Uncomment to start training
# train_gan(generator, discriminator, num_epochs=100, device=device)

"""Baseline Models for VSL Comparison (LSTM, GRU, Transformer)"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BASELINE 1: SIMPLE LSTM (minimal, no attention/augmentation)
# ============================================================
class SimpleLSTMBaseline(nn.Module):
    """Simplest LSTM baseline: 1-layer BiLSTM + fc"""
    def __init__(self, input_size, hidden_size=256, num_classes=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, 
                           bidirectional=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, seq_len, 2*hidden)
        # Use final hidden state
        x = lstm_out[:, -1, :]  # (batch, 2*hidden)
        x = self.fc(x)  # (batch, num_classes)
        return x


# ============================================================
# BASELINE 2: GRU BASELINE
# ============================================================
class GRUBaseline(nn.Module):
    """GRU baseline: 2-layer BiGRU + pooling + fc"""
    def __init__(self, input_size, hidden_size=256, num_classes=100):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True,
                         bidirectional=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, h_n = self.gru(x)  # (batch, seq_len, 2*hidden)
        # Average pooling over time
        x = gru_out.mean(dim=1)  # (batch, 2*hidden)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# BASELINE 3: TRANSFORMER BASELINE (SOTA reference)
# ============================================================
class TransformerBaseline(nn.Module):
    """Transformer baseline: positional encoding + transformer encoder + fc"""
    def __init__(self, input_size, hidden_size=256, num_heads=4, num_layers=2, num_classes=100):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_encoder = nn.Embedding(seq_len := 60, hidden_size)  # max seq length
        self.seq_len = seq_len
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len, input_features)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Project input to hidden size
        x = self.input_proj(x)  # (batch, seq_len, hidden)
        
        # Add positional encoding
        pos_ids = torch.arange(min(seq_len, self.seq_len), device=x.device).unsqueeze(0)
        pos_emb = self.pos_encoder(pos_ids)  # (1, seq_len, hidden)
        x = x + pos_emb
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, hidden)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, hidden)
        x = self.fc(x)  # (batch, num_classes)
        return x


# ============================================================
# IMPROVED: Your Model Architecture (CNN+BiLSTM+Attention+CosineClassifier)
# ============================================================
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for temporal sequences."""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out).mean(dim=1)  # pool over time


class CosineClassifier(nn.Module):
    """Cosine similarity classifier - much better for few-shot learning."""
    def __init__(self, in_features, num_classes, temperature=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, x):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return self.temperature * (x_norm @ w_norm.t())


class ImprovedSignModel(nn.Module):
    """Your model: CNN+BiLSTM+Attention+CosineClassifier"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        # Multi-scale temporal convolutions
        self.conv_k3 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(input_size, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(384)
        self.conv_drop = nn.Dropout(0.3)
        # BiLSTM
        self.lstm1 = nn.LSTM(384, 384, batch_first=True, bidirectional=True, 
                            num_layers=2, dropout=0.3)
        # Attention
        self.attention = MultiHeadAttention(768, num_heads=4)
        # Dense layers
        self.fc1 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        # Cosine classifier
        self.fc3 = CosineClassifier(256, num_classes)

    def forward(self, x):
        c = x.permute(0, 2, 1)  # (batch, features, seq)
        # Multi-scale conv
        c3 = F.gelu(self.conv_k3(c))
        c5 = F.gelu(self.conv_k5(c))
        c7 = F.gelu(self.conv_k7(c))
        c = torch.cat([c3, c5, c7], dim=1)  # 512
        c = F.gelu(self.bn1(c))
        c = F.gelu(self.bn2(self.conv2(c)))
        c = self.conv_drop(c)
        c = c.permute(0, 2, 1)  # (batch, seq, 384)
        lstm_out, _ = self.lstm1(c)
        attn_out = self.attention(lstm_out)
        x = F.gelu(self.bn3(self.fc1(attn_out)))
        x = self.drop1(x)
        x = F.gelu(self.bn4(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


# ============================================================
# ABLATION VARIANTS
# ============================================================
class AblationNoAugmentation(nn.Module):
    """Your model but without augmentation flag (tested in data loading)"""
    # Same as ImprovedSignModel, controlled by flag in training
    pass


class AblationNoAttention(nn.Module):
    """Your model without attention layer"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv_k3 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(input_size, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(384)
        self.conv_drop = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(384, 384, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=0.3)
        # Skip attention, use direct pooling
        self.fc1 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = CosineClassifier(256, num_classes)

    def forward(self, x):
        c = x.permute(0, 2, 1)
        c3 = F.gelu(self.conv_k3(c))
        c5 = F.gelu(self.conv_k5(c))
        c7 = F.gelu(self.conv_k7(c))
        c = torch.cat([c3, c5, c7], dim=1)
        c = F.gelu(self.bn1(c))
        c = F.gelu(self.bn2(self.conv2(c)))
        c = self.conv_drop(c)
        c = c.permute(0, 2, 1)
        lstm_out, _ = self.lstm1(c)
        # No attention: use mean pooling instead
        x = lstm_out.mean(dim=1)  # (batch, 768)
        x = F.gelu(self.bn3(self.fc1(x)))
        x = self.drop1(x)
        x = F.gelu(self.bn4(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x


class AblationNoCosineClassifier(nn.Module):
    """Your model but with softmax classifier instead of cosine"""
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.conv_k3 = nn.Conv1d(input_size, 256, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv_k7 = nn.Conv1d(input_size, 128, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = nn.Conv1d(512, 384, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(384)
        self.conv_drop = nn.Dropout(0.3)
        self.lstm1 = nn.LSTM(384, 384, batch_first=True, bidirectional=True,
                            num_layers=2, dropout=0.3)
        self.attention = MultiHeadAttention(768, num_heads=4)
        self.fc1 = nn.Linear(768, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.3)
        # Regular softmax classifier
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        c = x.permute(0, 2, 1)
        c3 = F.gelu(self.conv_k3(c))
        c5 = F.gelu(self.conv_k5(c))
        c7 = F.gelu(self.conv_k7(c))
        c = torch.cat([c3, c5, c7], dim=1)
        c = F.gelu(self.bn1(c))
        c = F.gelu(self.bn2(self.conv2(c)))
        c = self.conv_drop(c)
        c = c.permute(0, 2, 1)
        lstm_out, _ = self.lstm1(c)
        attn_out = self.attention(lstm_out)
        x = F.gelu(self.bn3(self.fc1(attn_out)))
        x = self.drop1(x)
        x = F.gelu(self.bn4(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)  # Regular softmax, not cosine
        return x


if __name__ == "__main__":
    # Test all models
    batch_size = 4
    seq_len = 60
    input_size = 4995
    num_classes = 100
    
    x = torch.randn(batch_size, seq_len, input_size)
    
    models_to_test = [
        ("SimpleLSTM", SimpleLSTMBaseline(input_size, num_classes=num_classes)),
        ("GRU", GRUBaseline(input_size, num_classes=num_classes)),
        ("Transformer", TransformerBaseline(input_size, num_classes=num_classes)),
        ("ImprovedSignModel", ImprovedSignModel(input_size, num_classes)),
        ("AblationNoAttention", AblationNoAttention(input_size, num_classes)),
        ("AblationNoCosineClassifier", AblationNoCosineClassifier(input_size, num_classes)),
    ]
    
    for name, model in models_to_test:
        try:
            y = model(x)
            print(f"✓ {name}: output shape {y.shape}")
        except Exception as e:
            print(f"✗ {name}: {e}")

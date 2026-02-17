"""
Deep Learning Model Architectures for ADR Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron with Dropout and Batch Normalization
    Simple but effective baseline deep learning model
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class EmbeddingMLP(nn.Module):
    """
    MLP with Embedding layers for categorical features
    Uses embeddings for high-cardinality features like drugs
    """
    def __init__(self, 
                 num_drugs, 
                 num_routes, 
                 num_dose_units,
                 embedding_dim=128,
                 numerical_features=8,
                 hidden_dims=[256, 128, 64],
                 dropout_rate=0.3):
        super(EmbeddingMLP, self).__init__()
        
        # Embedding layers
        self.drug_embedding = nn.Embedding(num_drugs, embedding_dim)
        self.route_embedding = nn.Embedding(num_routes, min(50, num_routes // 2))
        self.dose_unit_embedding = nn.Embedding(num_dose_units, min(50, num_dose_units // 2))
        
        # Calculate total input dimension
        embedding_total = embedding_dim + min(50, num_routes // 2) + min(50, num_dose_units // 2)
        total_input = embedding_total + numerical_features
        
        # MLP layers
        layers = []
        prev_dim = total_input
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, drug_idx, route_idx, dose_unit_idx, numerical_features):
        # Get embeddings
        drug_emb = self.drug_embedding(drug_idx)
        route_emb = self.route_embedding(route_idx)
        dose_unit_emb = self.dose_unit_embedding(dose_unit_idx)
        
        # Concatenate all features
        x = torch.cat([drug_emb, route_emb, dose_unit_emb, numerical_features], dim=1)
        
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual  # Skip connection
        out = F.relu(out)
        return out


class ResNetClassifier(nn.Module):
    """
    Residual Network for ADR prediction
    Uses skip connections for better gradient flow in deep networks
    """
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout_rate=0.3):
        super(ResNetClassifier, self).__init__()
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_layer(x)


class AttentionLayer(nn.Module):
    """Self-attention mechanism"""
    def __init__(self, dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5
    
    def forward(self, x):
        # x shape: (batch_size, dim)
        # Add sequence dimension
        x = x.unsqueeze(1)  # (batch_size, 1, dim)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        out = torch.matmul(attention_weights, V)
        return out.squeeze(1), attention_weights.squeeze(1)


class AttentionClassifier(nn.Module):
    """
    Attention-based classifier
    Uses self-attention to focus on important features
    """
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout_rate=0.3):
        super(AttentionClassifier, self).__init__()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dims[0])
        
        # MLP after attention
        layers = []
        prev_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.input_projection(x)
        x, attention_weights = self.attention(x)
        return self.mlp(x)


class DeepEnsemble(nn.Module):
    """
    Ensemble of multiple models
    Combines predictions from different architectures
    """
    def __init__(self, models, weights=None):
        super(DeepEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = torch.ones(len(models)) / len(models)
        else:
            self.weights = torch.tensor(weights)
    
    def forward(self, x):
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        
        # Weighted average
        stacked = torch.stack(predictions, dim=0)
        weights = self.weights.view(-1, 1, 1).to(x.device)
        weighted_pred = (stacked * weights).sum(dim=0)
        
        return weighted_pred


def get_model(model_type, **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: str, one of ['mlp', 'embedding', 'resnet', 'attention']
        **kwargs: model-specific arguments
    
    Returns:
        PyTorch model
    """
    if model_type == 'mlp':
        return MLPClassifier(**kwargs)
    elif model_type == 'embedding':
        return EmbeddingMLP(**kwargs)
    elif model_type == 'resnet':
        return ResNetClassifier(**kwargs)
    elif model_type == 'attention':
        return AttentionClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Focal Loss for handling class imbalance
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses training on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


if __name__ == "__main__":
    # Test models
    batch_size = 32
    input_dim = 13
    
    print("Testing MLPClassifier...")
    model = MLPClassifier(input_dim)
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("\nTesting ResNetClassifier...")
    model = ResNetClassifier(input_dim)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("\nTesting AttentionClassifier...")
    model = AttentionClassifier(input_dim)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    print("\nAll models working correctly!")
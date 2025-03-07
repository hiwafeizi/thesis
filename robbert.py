import torch
import json
import random
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Paths and configuration
feature_store_path = Path("data/features")
traits = ["evil", "trustworthy", "smart", "feminine"]
embeddings_file = feature_store_path / "word_embeddings.json"
variations_per_word = 1000
batch_size = 100
num_folds = 5
num_epochs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load embeddings
with open(embeddings_file, "r") as f:
    embeddings_dict = json.load(f)

def get_embedding(word):
    return torch.tensor(embeddings_dict[word], dtype=torch.float32)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input": self.features[idx], "labels": self.labels[idx]}

# Model structure
class TraitPredictor(nn.Module):
    def __init__(self, input_dim):
        super(TraitPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# Loop over each trait
for trait in traits:
    print(f"\n🚀 Training model for '{trait}' trait")
    
    # Load ratings data
    json_file = feature_store_path / f"word_ratings_{trait}.json"
    with open(json_file, "r") as f:
        word_data = json.load(f)

    all_words = list(word_data.keys())
    random.shuffle(all_words)

    # Create folds
    fold_size = len(all_words) // num_folds
    folds = [all_words[i * fold_size: (i + 1) * fold_size] for i in range(num_folds)]

    model = TraitPredictor(input_dim=768).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # K-Fold Training
    for fold in range(num_folds):
        test_words = folds[fold]
        train_words = [word for i in range(num_folds) if i != fold for word in folds[i]]
        print(f"🔄 Fold {fold+1}/{num_folds}: Train {len(train_words)} words, Test {len(test_words)} words")

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for batch_num in range(variations_per_word // batch_size):
                start_idx = batch_num * batch_size
                end_idx = start_idx + batch_size

                train_features, train_labels = [], []
                test_features, test_labels = [], []

                # Load train data
                for word in train_words:
                    embedding = get_embedding(word)
                    labels = word_data[word][start_idx:end_idx]
                    train_features.extend([embedding] * batch_size)
                    train_labels.extend(labels)

                # Load test data
                for word in test_words:
                    embedding = get_embedding(word)
                    labels = word_data[word][start_idx:end_idx]
                    test_features.extend([embedding] * batch_size)
                    test_labels.extend(labels)

                # Tensor conversion
                train_features = torch.stack(train_features)
                train_labels = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
                test_features = torch.stack(test_features)
                test_labels = torch.tensor(test_labels, dtype=torch.float32).unsqueeze(1)

                # Dataloaders
                train_loader = DataLoader(CustomDataset(train_features, train_labels), batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(CustomDataset(test_features, test_labels), batch_size=batch_size)

                # Training loop
                for batch in train_loader:
                    inputs, labels = batch["input"].to(device), batch["labels"].to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                print(f"Batch {batch_num+1}/{variations_per_word // batch_size} done")

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        print(f"✅ Completed Fold {fold+1} for '{trait}'")

    # Save trained model per trait
    model_save_path = feature_store_path / f"trained_{trait}_predictor.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Model for '{trait}' trait saved at {model_save_path}")

print("🎉 All traits training completed successfully!")

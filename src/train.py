import torch
from torch import nn, optim # nn - Contains neural network layers and loss functions. optim - Contains optimizers like Adam, SGD, etc.
from torch.utils.data import DataLoader # Helps load data in mini-batches for training.
from src.model import ShapeClassifier
from src.dataset import ShapeDataset

def train_model(data_path, model_path, epochs=30, batch_size=32):
    # Import the model and dataset
    dataset = ShapeDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Wraps the dataset in a DataLoader to enable batching and shuffling for each epoch.

    model = ShapeClassifier() # Initialize the neural network
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with learning rate 0.001
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss is suitable for multiclass classification

    for epoch in range(epochs):
        total_loss = 0 # Cumulative loss for this epoch
        correct = 0 # Number of correctly predicted samples
        for x, y in loader:  # Loop through each batch
            optimizer.zero_grad()  # Reset gradients from previous batch
            output = model(x)  # Forward pass through the model
            loss = criterion(output, y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights using gradients

            total_loss += loss.item() # Add batch loss to total
            correct += (output.argmax(1) == y).sum().item() # Count correct predictions

        accuracy = correct / len(dataset) # Accuracy over the full dataset
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)
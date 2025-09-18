## Imports
#---------------------------------------------------------------------------------------#

# Standard library
import argparse
import os
import sys
from pathlib import Path

# Third-party
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml

# Ensure the ASHA implementation module is discoverable when the script is executed directly
EXAMPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local
from asha_sweep import append_metric_value, get_fitness_scores, modify_current_runs

#---------------------------------------------------------------------------------------#


# Classes
#---------------------------------------------------------------------------------------#

# Regression neural network
class CaliforniaNN(nn.Module):
    def __init__(self, config):
        super(CaliforniaNN, self).__init__()
        layers = []
        input_size = 8  # California housing has 8 features
        
        # Define layers based on config
        for layer_size in [config["layer1_size"], config["layer2_size"], config["layer3_size"], config["layer4_size"]]:
            if layer_size > 0:
                layers.append(nn.Linear(input_size, layer_size))
                if config["batch_norm"]:
                    layers.append(nn.BatchNorm1d(layer_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config["dropout_rate"]))
                input_size = layer_size
        layers.append(nn.Linear(input_size, 1))  # Output layer for regression
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

#---------------------------------------------------------------------------------------#


# Functions
#---------------------------------------------------------------------------------------#

# Load and preprocess the dataset
def load_data(config, save_path):
    train_path = os.path.join(save_path, "data", "train.pkl")
    val_path = os.path.join(save_path, "data", "val.pkl")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Loading preprocessed datasets from disk...")

        # Load existing data
        with open(train_path, "rb") as f:
            train_data = pickle.load(f)
        with open(val_path, "rb") as f:
            val_data = pickle.load(f)
        train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)
    else:
        print("Preprocessing California housing dataset and saving to disk...")

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        
        # Fetch and preprocess the dataset
        data = fetch_california_housing()
        X, y = data.data, data.target
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Save datasets
        with open(train_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(val_path, "wb") as f:
            pickle.dump(val_dataset, f)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    return train_loader, val_loader

# Training loop
def train_model(model, train_loader, device, config, num_epochs):
    print(f"Starting training for {num_epochs} epochs using {config['optimizer_type']} optimizer on {device}.")

    model.to(device)

    criterion = nn.MSELoss()
    criterion.to(device)

    if config["optimizer_type"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["lr"], momentum=config.get("momentum", 0), weight_decay=config["weight_decay"])
    elif config["optimizer_type"] == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        optimizer = None

    # Assert that an optimizer has been initialized
    assert optimizer is not None, f"Invalid optimizer type: {config['optimizer_type']}. Supported types are 'adam', 'sgd', 'rmsprop', and 'adagrad'."
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        train_loss /= len(train_loader.dataset)

# Validation
def validate_model(model, val_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f"Validation Loss: {val_loss:.4f}")

    return val_loss

# Save model function
def save_model(model, path):
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), path)

# Load model function
def load_model(path, config):
    print(f"Loading model from {path}")
    model = CaliforniaNN(config)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Main function
def main(run_dir, epochs, gpu_number):

    # Intialize paths
    sweep_path = os.path.dirname(os.path.dirname(os.path.dirname(run_dir)))
    config_name = os.path.basename(run_dir)
    config_path = os.path.join(run_dir, "config_params.yaml")
    trained_models_dir = os.path.join(run_dir, "trained_models")
    current_rung = len(get_fitness_scores(sweep_path, config_name))

    # Load configuration
    print("Loading configuration...")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # Set up device
    print(f"Using GPU {gpu_number}")
    device = torch.device(f"cuda:{gpu_number}" if torch.cuda.is_available() and gpu_number < torch.cuda.device_count() else "cpu")

    # Load data
    train_loader, val_loader = load_data(config, sweep_path)
    print("Training completed, evaluating on validation set...")
    
    # Initialize model
    model = CaliforniaNN(config)

    # Load the model if we are not on the first rung
    if current_rung > 0:
        model = load_model(
            os.path.join(
                trained_models_dir, 
                f'model_after_training_on_rung_{current_rung-1}.pt'
            ), 
            config
        )
    
    # Train the model
    train_model(model, train_loader, device, config, epochs)
    
    # Save the trained model
    model_save_path = os.path.join(
        trained_models_dir, 
        f'model_after_training_on_rung_{current_rung}.pt'
    )
    save_model(model, model_save_path)
    
    # Validate the model at the end
    val_loss = validate_model(model, val_loader, device)

    # Calculate fitness score and return
    fitness_score = -val_loss

    print(f"Final fitness score: {fitness_score:.4f}")

    return fitness_score

#---------------------------------------------------------------------------------------#


# Entry point
#---------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a California Housing regression model.")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--resources_to_add", type=str, required=True, help="Additional resources to add.")
    parser.add_argument("--gpu", type=str, required=True, help="GPU index to use (default: 0).")
    args = parser.parse_args()

    # Generate path names
    sweep_path = os.path.dirname(os.path.dirname(os.path.dirname(args.run_dir)))
    config_name = os.path.basename(args.run_dir)

    # Indicate run on gpu in ledger
    modify_current_runs(sweep_path, config_name, args.gpu, add=True)
    
    # Train for N resources and return fitness score
    fitness_score = main(
        args.run_dir,
        int(args.resources_to_add),
        int(args.gpu),
    )

    # Log rungs fitness score in ledger
    append_metric_value(
        sweep_path, 
        config_name, 
        "fitness_scores", fitness_score
    )

    # Indicate run on gpu is over in ledger
    modify_current_runs(sweep_path, config_name, add=False)

    # Exit the script
    sys.exit(0)

#---------------------------------------------------------------------------------------#

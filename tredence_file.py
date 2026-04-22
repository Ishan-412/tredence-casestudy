"""
Self-Pruning Neural Network for CIFAR-10 Classification
========================================================
This script implements a Multi-Layer Perceptron (MLP) with learnable gate
parameters that allow the network to automatically prune (zero-out) unimportant
weights during training. A sparsity-inducing regularisation term is added to
the loss function so the model learns to deactivate unnecessary connections,
yielding a sparser — and potentially more efficient — network.

Key concepts:
  - **Gated weights**: Each weight has a companion gate score passed through a
    sigmoid, producing a value in [0, 1]. Multiplying the weight by its gate
    lets the optimiser "turn off" weights by driving the gate toward 0.
  - **Sparsity loss**: The L1 norm of all gate values is added to the
    classification loss, weighted by a tuneable hyper-parameter `lambda_sparse`.
  - **Trade-off analysis**: Multiple lambda values are tested to visualise how
    increasing sparsity pressure affects classification accuracy.
"""

# ──────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed=42):
    """
    Fix all random-number-generator seeds so that results are reproducible
    across runs, including CPU and GPU operations.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Automatically select GPU if available; otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# Custom Prunable Linear Layer
# ──────────────────────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    A fully-connected (linear) layer augmented with learnable gate scores.

    For every weight w_ij the layer maintains a corresponding gate score g_ij.
    During the forward pass the effective weight is:
        w_ij_eff = w_ij * sigmoid(g_ij)
    When sigmoid(g_ij) ≈ 0 the connection is effectively pruned.
    """

    def __init__(self, in_features, out_features):
        """
        Args:
            in_features  (int): Number of input neurons.
            out_features (int): Number of output neurons.
        """
        super().__init__()

        # Standard weight matrix, initialised with small random values.
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)

        # Bias vector, initialised to zero.
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores — same shape as the weight matrix.  These are raw
        # (pre-sigmoid) values so the optimiser can update them freely in
        # (-∞, +∞) while the actual gate values remain in [0, 1].
        self.gate_scores = nn.Parameter(torch.randn_like(self.weight))

    def forward(self, x):
        """
        Forward pass: apply sigmoid-gated weights to the input.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output of shape (batch_size, out_features).
        """
        # Squash gate scores into [0, 1] via sigmoid.
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise multiply: gates close to 0 effectively prune that weight.
        pruned_weight = self.weight * gates

        # Standard linear transformation: x @ W^T + b.
        return torch.nn.functional.linear(x, pruned_weight, self.bias)

    def get_gate_values(self):
        """Return the current gate activations (sigmoid of raw gate scores)."""
        return torch.sigmoid(self.gate_scores)


# ──────────────────────────────────────────────────────────────
# Self-Pruning MLP Architecture
# ──────────────────────────────────────────────────────────────
class PrunableMLP(nn.Module):
    """
    A three-layer MLP for CIFAR-10 (10-class classification) where every
    linear layer is a PrunableLinear, enabling automatic weight pruning.

    Architecture:
        Input (3072 = 32×32×3) → 512 → 256 → 10 (logits)
    """

    def __init__(self):
        super().__init__()

        # Flatten spatial dimensions of CIFAR-10 images (32×32×3 → 3072).
        self.flatten = nn.Flatten()

        # Three prunable fully-connected layers.
        self.fc1 = PrunableLinear(32*32*3, 512)   # Hidden layer 1
        self.fc2 = PrunableLinear(512, 256)        # Hidden layer 2
        self.fc3 = PrunableLinear(256, 10)         # Output layer (10 classes)

        # Activation function applied after hidden layers.
        self.relu = nn.ReLU()

        # Convenient list for iterating over all prunable layers.
        self.prunable_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        """
        Forward pass through the MLP.

        Args:
            x (Tensor): Batch of CIFAR-10 images, shape (B, 3, 32, 32).

        Returns:
            Tensor: Raw logits of shape (B, 10).
        """
        x = self.flatten(x)           # (B, 3072)
        x = self.relu(self.fc1(x))    # (B, 512)
        x = self.relu(self.fc2(x))    # (B, 256)
        x = self.fc3(x)               # (B, 10)  — no activation; logits
        return x

    def sparsity_loss(self):
        """
        Compute the L1 sparsity penalty over all gate values.

        Summing |sigmoid(g)| for every gate encourages the optimiser to push
        gate values toward 0, effectively pruning connections.

        Returns:
            Tensor (scalar): Total L1 gate penalty.
        """
        loss = 0
        for layer in self.prunable_layers:
            loss += torch.sum(torch.abs(layer.get_gate_values()))
        return loss

    def compute_sparsity(self, threshold=1e-2):
        """
        Measure what percentage of gates are "effectively zero" (below a
        given threshold).

        Args:
            threshold (float): Gate values below this are considered pruned.

        Returns:
            float: Sparsity percentage (0–100).
        """
        total = 0
        zero = 0

        for layer in self.prunable_layers:
            gates = layer.get_gate_values()
            total += gates.numel()                       # Total number of gates
            zero += (gates < threshold).sum().item()     # Count of pruned gates

        return 100 * zero / total


# ──────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ──────────────────────────────────────────────────────────────

# Standard CIFAR-10 preprocessing: convert to tensor and normalise each
# channel to zero-mean, unit-range (pixel values mapped from [0,1] to [-1,1]).
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download (if needed) and wrap CIFAR-10 in DataLoaders for batched iteration.
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────
def train_model(lambda_sparse, epochs=5):
    """
    Train a fresh PrunableMLP with the specified sparsity weight.

    The total loss per batch is:
        L_total = CrossEntropy(predictions, labels)
                  + lambda_sparse * Σ|gate_values|

    Args:
        lambda_sparse (float): Weight of the sparsity penalty.
        epochs        (int):   Number of full passes over the training set.

    Returns:
        model        (PrunableMLP): The trained model.
        epoch_losses (list[float]): Cumulative loss for each epoch.
    """
    model = PrunableMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Classification loss (cross-entropy)
            cls_loss = criterion(outputs, labels)

            # Sparsity regularisation (L1 on gate activations)
            sparse_loss = model.sparsity_loss()

            # Combined objective
            loss = cls_loss + lambda_sparse * sparse_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_losses.append(total_loss)
        print(f"Lambda {lambda_sparse} | Epoch {epoch+1} | Loss: {total_loss:.4f}")

    return model, epoch_losses


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
def evaluate(model):
    """
    Evaluate the model on the CIFAR-10 test set.

    Args:
        model (PrunableMLP): A trained model.

    Returns:
        accuracy (float): Classification accuracy (%).
        sparsity (float): Percentage of gates below the pruning threshold.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    sparsity = model.compute_sparsity()

    return accuracy, sparsity


# ──────────────────────────────────────────────────────────────
# Experiment: Sweep Over Multiple Lambda Values
# ──────────────────────────────────────────────────────────────

# Each lambda controls how aggressively the network prunes weights.
# Larger lambda → stronger sparsity pressure → sparser but potentially
# less accurate model.
lambdas = [1e-6, 1e-5, 1e-4]
results = []          # Stores (lambda, accuracy, sparsity) tuples
models = {}           # Trained model keyed by lambda
loss_history = {}     # Per-epoch loss history keyed by lambda

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")
    model, losses = train_model(lam, epochs=5)

    acc, sparsity = evaluate(model)
    results.append((lam, acc, sparsity))
    models[lam] = model
    loss_history[lam] = losses

    print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")


# ──────────────────────────────────────────────────────────────
# Visualisation 1: Training Loss Curves
# ──────────────────────────────────────────────────────────────
# Plot the training loss per epoch for each lambda to compare convergence
# behaviour under different sparsity pressures.
plt.figure()
for lam in lambdas:
    plt.plot(loss_history[lam], label=f"λ={lam}")

plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# ──────────────────────────────────────────────────────────────
# Visualisation 2: Sparsity-vs-Accuracy Trade-off
# ──────────────────────────────────────────────────────────────
# This plot reveals the Pareto-like frontier: how much accuracy is
# sacrificed as the network becomes sparser.
lams = [r[0] for r in results]
accs = [r[1] for r in results]
sparsities = [r[2] for r in results]

plt.figure()
plt.plot(sparsities, accs, marker='o')

# Annotate each point with its lambda value for reference.
for i, lam in enumerate(lams):
    plt.text(sparsities[i], accs[i], f"{lam}")

plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Sparsity vs Accuracy Tradeoff")
plt.show()


# ──────────────────────────────────────────────────────────────
# Visualisation 3: Gate Value Distribution
# ──────────────────────────────────────────────────────────────
def plot_gate_distribution(model):
    """
    Plot a histogram of all gate values in the model.

    A well-pruned model will show a bimodal distribution: many gates
    clustered near 0 (pruned) and the rest near 1 (active).

    Args:
        model (PrunableMLP): A trained model whose gate distribution to plot.
    """
    all_gates = []

    for layer in model.prunable_layers:
        gates = layer.get_gate_values().detach().cpu().numpy().flatten()
        all_gates.extend(gates)

    plt.figure()
    plt.hist(all_gates, bins=100)
    plt.title("Gate Value Distribution (Best Model)")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.show()

# Plot gate distribution for the model trained with the strongest
# sparsity pressure (last / largest lambda).
plot_gate_distribution(models[lambdas[-1]])


# ──────────────────────────────────────────────────────────────
# Summary Table
# ──────────────────────────────────────────────────────────────
# Print a final comparison of all lambda configurations.
print("\nFinal Results:")
print("Lambda | Accuracy | Sparsity (%)")
for r in results:
    print(f"{r[0]} | {r[1]:.2f}% | {r[2]:.2f}%")
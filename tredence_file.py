import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn_like(self.weight))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weight = self.weight * gates
        return torch.nn.functional.linear(x, pruned_weight, self.bias)

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores)



class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc1 = PrunableLinear(32*32*3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()

        self.prunable_layers = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sparsity_loss(self):
        loss = 0
        for layer in self.prunable_layers:
            loss += torch.sum(torch.abs(layer.get_gate_values()))
        return loss

    def compute_sparsity(self, threshold=1e-2):
        total = 0
        zero = 0

        for layer in self.prunable_layers:
            gates = layer.get_gate_values()
            total += gates.numel()
            zero += (gates < threshold).sum().item()

        return 100 * zero / total



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)



def train_model(lambda_sparse, epochs=5):
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

            outputs = model(images)
            cls_loss = criterion(outputs, labels)
            sparse_loss = model.sparsity_loss()

            loss = cls_loss + lambda_sparse * sparse_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_losses.append(total_loss)
        print(f"Lambda {lambda_sparse} | Epoch {epoch+1} | Loss: {total_loss:.4f}")

    return model, epoch_losses


def evaluate(model):
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



lambdas = [1e-6, 1e-5, 1e-4]
results = []
models = {}
loss_history = {}

for lam in lambdas:
    print(f"\nTraining with lambda = {lam}")
    model, losses = train_model(lam, epochs=5)

    acc, sparsity = evaluate(model)
    results.append((lam, acc, sparsity))
    models[lam] = model
    loss_history[lam] = losses

    print(f"Lambda: {lam} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")



plt.figure()
for lam in lambdas:
    plt.plot(loss_history[lam], label=f"λ={lam}")

plt.title("Training Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

lams = [r[0] for r in results]
accs = [r[1] for r in results]
sparsities = [r[2] for r in results]

plt.figure()
plt.plot(sparsities, accs, marker='o')

for i, lam in enumerate(lams):
    plt.text(sparsities[i], accs[i], f"{lam}")

plt.xlabel("Sparsity (%)")
plt.ylabel("Accuracy (%)")
plt.title("Sparsity vs Accuracy Tradeoff")
plt.show()


def plot_gate_distribution(model):
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

plot_gate_distribution(models[lambdas[-1]])


print("\nFinal Results:")
print("Lambda | Accuracy | Sparsity (%)")
for r in results:
    print(f"{r[0]} | {r[1]:.2f}% | {r[2]:.2f}%")
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ==========================================
# 1. DATA PREPARATION
# ==========================================
def load_data():
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize to [0, 1] range (Input X)
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

    # Filter only digits 0 and 8
    # Hypothesis H0: Digit 0 (Label 0)
    # Hypothesis H1: Digit 8 (Label 1)
    train_mask = np.isin(y_train, [0, 8])
    test_mask = np.isin(y_test, [0, 8])

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Remap labels: 0 -> 0, 8 -> 1 (Binary Classification)
    y_train = np.where(y_train == 0, 0, 1).reshape(-1, 1)
    y_test = np.where(y_test == 0, 0, 1).reshape(-1, 1)

    print(f"Training Samples: {x_train.shape[0]} (0s and 8s)")
    print(f"Test Samples: {x_test.shape[0]} (0s and 8s)")
    
    return x_train, y_train, x_test, y_test

# ==========================================
# 2. NEURAL NETWORK (THE TOOL)
# ==========================================
class SimpleNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with small random values (He initialization approximation)
        # Theta = {W1, b1, W2, b2}
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        # Activation function phi(u)
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        # Forward pass: u(x, theta)
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1) # Hidden layer output
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2) # Output layer (probability)
        return self.a2

    def backward(self, x, y, learning_rate):
        # ==========================================
        # 3. THE BRIDGE (COST FUNCTION DERIVATIVE)
        # ==========================================
        # We use Cross-Entropy Loss: J = -[y*log(a2) + (1-y)*log(1-a2)]
        # The derivative of Cost w.r.t final output z2 is simply (a2 - y)
        # This is the "Magic" of Cross-Entropy + Sigmoid.
        
        m = x.shape[0] # Batch size
        
        # Gradient of Loss w.r.t Output (dZ2)
        dZ2 = self.a2 - y 
        
        # Gradients for W2, b2
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backpropagate to Hidden Layer
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.a1 * (1 - self.a1)) # Derivative of sigmoid
        
        # Gradients for W1, b1
        dW1 = np.dot(x.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # ==========================================
        # 4. LEARNING ALGORITHM (SGD UPDATE)
        # ==========================================
        # theta_t = theta_{t-1} - mu * gradient
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def compute_loss(self, y_true, y_pred):
        # Binary Cross Entropy Cost
        m = y_true.shape[0]
        # Clip to avoid log(0) error
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Setup
x_train, y_train, x_test, y_test = load_data()
model = SimpleNeuralNet(input_size=784, hidden_size=64, output_size=1)

# Hyperparameters
learning_rate = 0.1
epochs = 50
batch_size = 32
loss_history = []

print(f"\nTraining for {epochs} epochs using SGD...")

# 2. Training Loop
for epoch in range(epochs):
    # Shuffle data (Stochastic part of SGD)
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    epoch_loss = 0
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward
        predictions = model.forward(x_batch)
        
        # Backward (Update weights)
        model.backward(x_batch, y_batch, learning_rate)
        
        # Track Loss
        loss = model.compute_loss(y_batch, predictions)
        epoch_loss += loss
        
    avg_loss = epoch_loss / (x_train.shape[0] / batch_size)
    loss_history.append(avg_loss)
    
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

# ==========================================
# 5. EVALUATION (ROC CURVE)
# ==========================================
print("\nEvaluating...")
y_pred_test = model.forward(x_test)

# Calculate ROC Curve metrics manually
thresholds = np.linspace(0, 1, 100)
tpr_list = [] # True Positive Rate (Detection Prob)
fpr_list = [] # False Positive Rate (False Alarm Prob)

for thresh in thresholds:
    # Decision Rule: u(x) > threshold
    y_decision = (y_pred_test >= thresh).astype(int)
    
    # Confusion Matrix Elements
    TP = np.sum((y_decision == 1) & (y_test == 1)) # Correctly found 8
    FP = np.sum((y_decision == 1) & (y_test == 0)) # Mistook 0 for 8
    FN = np.sum((y_decision == 0) & (y_test == 1)) # Missed 8
    TN = np.sum((y_decision == 0) & (y_test == 0)) # Correctly found 0
    
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    tpr_list.append(TPR)
    fpr_list.append(FPR)

# ==========================================
# VISUALIZATION
# ==========================================
plt.figure(figsize=(12, 5))

# Plot 1: Learning Curve
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Cross-Entropy Loss')
plt.title('Learning Curve (Cost Minimization)')
plt.xlabel('Epochs')
plt.ylabel('Cost J(theta)')
plt.grid(True)
plt.legend()

# Plot 2: ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Alarm Rate (P_FA)')
plt.ylabel('Detection Rate (P_D)')
plt.title('ROC Curve (Approximating Neyman-Pearson)')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()

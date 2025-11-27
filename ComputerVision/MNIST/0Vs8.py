'''
- Βιβλιοθήκες που χρειάζονται:
pip install tensorflow
pip install matplotlib
pip install numpy
'''

import numpy as np
import matplotlib.pyplot as plt

# ΜΑΣ ΤΑ ΕΧΟΥΝΕ ΖΑΛ... ΕΝΑ TENSORFLOW ΚΑΙ ΕΝΑ GYM!!!
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 2 = no INFO/WARNING
from tensorflow.keras.datasets import mnist



# Φόρτωσε κατάλληλα τα δεδομένα από το MNIST!
def load_data() -> tuple:
    print('Φόρτωση δεδομένων MNIST...')
    ((x_train, y_train), (x_test, y_test)) = mnist.load_data()
    # x -> Εικόνες [(60 000, 28, 28)]
    # y -> Ετικέτες (0-9) [(60 000,) - 1D array] 

    # Κανονικοποίηση των pixel της εισόδου στο διάστημα [0, 1]
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0 # (60 000, 784)
    x_test  = x_test.reshape(-1,  784).astype('float32') / 255.0

    # Κρατάμε μόνο τα ψηφία 0 και 8
    # Hypothesis H0: Ψηφίο 0 (Label 0)
    # Hypothesis H1: Ψηφίο 8 (Label 1)
    train_mask = np.isin(y_train, [0, 8])
    test_mask  = np.isin(y_test,  [0, 8])

    (x_train, y_train) = (x_train[train_mask], y_train[train_mask])
    (x_test,  y_test)  = (x_test[test_mask],   y_test[test_mask])

    # Remap labels: 0 -> 0, 8 -> 1 (Binary Classification)!
    y_train = np.where(y_train == 0, 0, 1).reshape(-1, 1)
    y_test  = np.where(y_test == 0,  0, 1).reshape(-1, 1)
    # reshape(-1, 1) -> διάνυσμα στήλης

    print(f'Δεδομένα training: {x_train.shape[0]}')
    print(f'Δεδομένα test:     {x_test.shape[0]}')
    
    return (x_train, y_train, x_test, y_test);



class NeuralNet:
    def __init__(self, input_size:  int, hidden_size: int, output_size: int) -> None:
        # Τυχαία αρχικοποίηση των weights και biases!
        # θ = {W1, b1, W2, b2}
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        return;

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z)); # Activation function φ(u)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Forward pass: u(x, θ)
        self.z1 = x @ self.W1 + self.b1 # @ -> matrix multiplication
        self.a1 = self.sigmoid(self.z1) # Hidden layer output

        self.z2 = self.a1 @ self.W2 + self.b2 # ή np.dot(... , ...)!
        self.a2 = self.sigmoid(self.z2) # Output layer (πιθανότητα)

        return self.a2;

    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        # Cross-Entropy Loss: J = -[y*log(a2) + (1-y)*log(1-a2)]
        
        m = x.shape[0] # Batch size
        
        # Gradient of Loss w.r.t Output (dZ2)
        dZ2 = self.a2 - y 
        
        # Gradients for W2, b2
        dW2 = (self.a1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backpropagate to Hidden Layer
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (self.a1 * (1 - self.a1)) # Παράγωγος sigmoid
        
        # Gradients for W1, b1
        dW1 = (x.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # --- LEARNING ALGORITHM (SGD UPDATE) --- #
        # θ_t = θ_{t-1} - μ * gradient
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        return;

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Binary Cross Entropy Cost
        m = y_true.shape[0]

        # Clip to avoid log(0) error!!!
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss   = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

        return loss;



# ===< MAIN SCRIPT >=== #
(x_train, y_train, x_test, y_test) = load_data()
model = NeuralNet(input_size = 784, hidden_size = 64, output_size = 1)

# Hyperparameters
learning_rate = 0.1
epochs        = 50
batch_size    = 32
loss_history  = []

# Training Loop
print(f'\nTraining για {epochs} epochs με SGD...')
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

        # Backward (ενημέρωση weights)
        model.backward(x_batch, y_batch, learning_rate)
        
        # Track Loss
        loss        = model.compute_loss(y_batch, predictions)
        epoch_loss += loss
        
    avg_loss = epoch_loss / (x_train.shape[0] / batch_size)
    loss_history.append(avg_loss)
    
    if epoch % 5 == 0: print(f'Epoch {epoch}: Loss = {avg_loss:.4f}')



# ===< EVALUATION (ROC CURVE) >=== #
print('\nEvaluating...')
y_pred_test = model.forward(x_test)

thresholds = np.linspace(0, 1, 100) # Ρίψη νομίσματος
tpr_list = [] # True Positive Rate (Detection Prob)
fpr_list = [] # False Positive Rate (False Alarm Prob)

for thresh in thresholds:
    # Decision Rule: u(x) > threshold
    y_decision = (y_pred_test >= thresh).astype(int)
    
    # Confusion Matrix Elements
    TP = np.sum((y_decision == 1) & (y_test == 1)) # Μάντεψε σωστά 8
    FP = np.sum((y_decision == 1) & (y_test == 0)) # Μπέρδεψε 0 με 8
    FN = np.sum((y_decision == 0) & (y_test == 1)) # Μπέρδεψε 8 με 0
    TN = np.sum((y_decision == 0) & (y_test == 0)) # Μάντεψε σωστά 0

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    
    tpr_list.append(TPR)
    fpr_list.append(FPR)

# ===< VISUALIZATION >=== #
plt.figure(figsize = (12, 5))

# Learning Curve
plt.subplot(1, 2, 1)
plt.plot(loss_history, label = 'Cross-Entropy Loss')
plt.title('Learning Curve (Cost Minimization)')
plt.xlabel('Epochs')
plt.ylabel('Cost J(theta)')
plt.grid(True)
plt.legend()

# ROC Curve
plt.subplot(1, 2, 2)
plt.plot(fpr_list, tpr_list, color = 'darkorange', lw = 2, label = 'ROC curve')
plt.plot([0, 1], [0, 1], color = 'navy', lw = 2, linestyle = '--') # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Alarm Rate (P_FA)')
plt.ylabel('Detection Rate (P_D)')
plt.title('ROC Curve (Approximating Neyman-Pearson)')
plt.legend(loc = 'lower right')
plt.grid(True)

plt.tight_layout()
plt.show()



# ===< INTERACTIVE PAINT APPLICATION >=== #
import tkinter as tk
def launch_paint_app(model: NeuralNet) -> None:
    CANVAS_SIZE = 280 # 280x280 pixels
    GRID_SIZE   = 28  # 28x28 όπως το MNIST!
    CELL_SIZE   = CANVAS_SIZE // GRID_SIZE

    # Πίνακας 28x28 που θα δίνουμε στο νευρωνικό (τιμή 0..1)
    image_array = np.zeros((GRID_SIZE, GRID_SIZE), dtype = np.float32)

    # Δημιουργία παραθύρου Tk...
    root = tk.Tk()
    root.title('Ζωγράφισε 0 ή 8!')

    canvas = tk.Canvas(root, width = CANVAS_SIZE, height = CANVAS_SIZE, bg = 'black')
    canvas.pack(padx = 10, pady = 10)

    # Label για πρόβλεψη
    prediction_var   = tk.StringVar(value = 'Ζωγράφισε ένα ψηφίο (0 ή 8)')
    prediction_label = tk.Label(root, textvariable = prediction_var, font = ('Arial', 14))
    prediction_label.pack(pady = 5)

    # Συνάρτηση για καθαρισμό του Canvas!
    def clear_canvas():
        canvas.delete('all') # Καθαρίζει το Canvas
        image_array[:] = 0.0 # Μηδενίζει τον πίνακα 28x28
        prediction_var.set('Ζωγράφισε ένα ψηφίο (0 ή 8)') # Επαναφέρει το label

        return;

    clear_button = tk.Button(root, text = 'Επαναφορά', command = clear_canvas)
    clear_button.pack(pady = 5)

    # Συνάρτηση που ενημερώνει την πρόβλεψη του μοντέλου
    def update_prediction():
        # Μετατρέπουμε το 28x28 σε (1, 784)
        x_input         = image_array.reshape(1, -1) # ήδη 0..1!
        prob_8          = float(model.forward(x_input)[0, 0])
        predicted_label = 8 if prob_8 >= 0.5 else 0
        prediction_var.set(f'Το μοντέλο επιστρέφει: {predicted_label}  (p(8) = {prob_8:.2f})')

        return;

    # Συνάρτηση ζωγραφικής με το mouse
    def draw(event):
        (x, y) = (event.x, event.y)
        if (0 <= x < CANVAS_SIZE) and (0 <= y < CANVAS_SIZE):
            j = x // CELL_SIZE # column
            i = y // CELL_SIZE # row

            # Κάνουμε λίγο πιο 'παχύ' το πινέλο: 3x3 γύρω από το σημείο
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < GRID_SIZE and 0 <= jj < GRID_SIZE:
                        image_array[ii, jj] = 1.0
                        x0 = jj * CELL_SIZE
                        y0 = ii * CELL_SIZE
                        x1 = x0 + CELL_SIZE
                        y1 = y0 + CELL_SIZE
                        canvas.create_rectangle(
                            x0, y0, x1, y1,
                            fill = 'white', outline = 'white'
                        )
            # Μετά από κάθε stroke, ενημέρωσε πρόβλεψη
            update_prediction()

        return;

    # Ζωγραφίζουμε κρατώντας πατημένο το αριστερό κουμπί
    canvas.bind('<B1-Motion>', draw)

    root.mainloop()

    return;

# Εκκίνηση του Tk παραθύρου μετά τα plots...
launch_paint_app(model)

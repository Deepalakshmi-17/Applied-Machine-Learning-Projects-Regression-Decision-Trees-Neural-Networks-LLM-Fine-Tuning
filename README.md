Task 1 - House Price Prediction using Linear Regression:

This task involves predicting property prices in Pune using basic linear regression. The dataset included features like total square footage, number of bedrooms/bathrooms, and location. The model pipeline included data cleaning (e.g., handling missing values, feature extraction like bhk), encoding categorical variables, and fitting a regression model. The final model achieved a good R² score and low RMSE, visualized with an actual vs predicted scatter plot.

python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Pune_House_Data.csv')
df.drop('society', axis=1, inplace=True)
df['bath'].fillna(df['bath'].median(), inplace=True)
df['balcony'].fillna(df['balcony'].median(), inplace=True)
df['bhk'] = df['size'].str.extract('(\d+)').astype(float)
df.drop('size', axis=1, inplace=True)

def convert_sqft(x):
    try:
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        else:
            return float(x)
    except:
        return np.nan

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df.dropna(subset=['total_sqft'], inplace=True)
df = pd.get_dummies(df, columns=['area_type', 'availability', 'site_location'], drop_first=True)

X = df.drop('price', axis=1)
y = df['price']
X.fillna(X.median(), inplace=True)
y.fillna(y.median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
print("R² Score:", r2)
print("RMSE:", rmse)

plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted House Prices in Pune")
plt.grid(True)
plt.show()


Task 2 - Heart Disease Classification using Decision Trees:
In this task, I worked on a classification problem using the UCI Heart Disease dataset. The goal was to predict whether a patient is likely to have heart disease based on features such as age, sex, blood pressure, cholesterol, etc. I performed data preprocessing, handled missing values, applied one-hot encoding for categorical features, and trained a Decision Tree Classifier. The model was further simplified by pruning to avoid overfitting. Evaluation was done using accuracy, classification report, and confusion matrix.

python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

column_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df = pd.read_csv('processed.cleveland.data', names=column_names)
df.replace('?', pd.NA, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)

X = df.drop('target', axis=1)
y = df['target']
y = y.apply(lambda x: 1 if x > 0 else 0)

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(20,10))
plot_tree(tree, filled=True, feature_names=X.columns, class_names=["No Disease", "Disease"])
plt.show()

conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


Task 3 - Image Classification using Neural Networks (Fashion MNIST) 

 python
 In this task, I built a basic neural network using TensorFlow/Keras to classify images of clothing items from the Fashion MNIST dataset. The dataset consists of grayscale images across 10 fashion categories like shirts, shoes, and bags. The process involved normalizing image data, defining a simple feedforward neural network, compiling with appropriate loss and optimizer, and evaluating model performance through training/testing accuracy and loss curves. This task gave hands-on experience with deep learning basics.

 import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
 
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
 
X_train = X_train / 255.0
X_test = X_test / 255.0

 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

 
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
 
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)
 
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

 
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


Task 4 – Improving a Failing Language Model (Transformer):
In this task, I was given a poorly performing transformer-based language model in a Colab notebook. The model was showing high loss and generating incoherent text. My job was to tune its hyperparameters, adjust training settings, and ensure the loss decreases steadily. I improved model performance by changing the learning rate, batch size, and sequence length, and also optimized the model architecture to make training more stable and results more meaningful.
This task helped me understand the sensitivity of LLMs to training configuration and how hyperparameter tuning significantly affects performance.

python
import torch
import torch.nn as nn
from torch.nn import functional as F
import random
 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 128
batch_size = 64
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            _, val_loss = model(xb, yb)
        print(f"Step {iter}: val loss = {val_loss.item():.4f}")
        model.train()

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200)
print(decode(generated[0].tolist()))

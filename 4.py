# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)

# Load the credit card dataset
df = pd.read_csv('/content/creditcard.csv')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nClass Distribution:")
print(df['Class'].value_counts())
print("\nPercentage of Fraudulent Transactions:",
      round(df['Class'].value_counts()[1] / len(df) * 100, 2), '%')
     

# Normalize the Time and Amount columns using StandardScaler
scaler = StandardScaler()

# Scale the 'Amount' and 'Time' columns
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

# Drop the original 'Time' and 'Amount' columns
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# Reorder columns to have scaled features at the beginning
scaled_amount = df['scaled_amount']
scaled_time = df['scaled_time']

df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', scaled_amount)
df.insert(1, 'scaled_time', scaled_time)

print("Dataset after scaling:")
print(df.head())

# Separate normal and fraudulent transactions
normal_df = df[df['Class'] == 0]
fraud_df = df[df['Class'] == 1]

print("Number of Normal Transactions:", len(normal_df))
print("Number of Fraudulent Transactions:", len(fraud_df))

# Use only normal transactions for training
X_train = normal_df.drop(['Class'], axis=1).values
y_train = normal_df['Class'].values

print("\nTraining Data Shape:", X_train.shape)

# Split the entire dataset into train and test
X = df.drop(['Class'], axis=1).values
y = df['Class'].values

# Split data: 80% train, 20% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use only normal transactions from training set
X_train = X_train_full[y_train_full == 0]
X_test_normal = X_test[y_test == 0]
X_test_fraud = X_test[y_test == 1]

print("Training Data (Normal only):", X_train.shape)
print("Test Data:", X_test.shape)
print("Test Normal:", X_test_normal.shape)
print("Test Fraud:", X_test_fraud.shape)


# Define the input dimension
input_dim = X_train.shape[1]

# Set encoding dimensions
encoding_dim_1 = 14
encoding_dim_2 = 7
encoding_dim_3 = 4

# Learning rate and epochs
learning_rate = 0.001
epochs = 50
batch_size = 32

print(f"Input Dimension: {input_dim}")
print(f"Encoding Dimensions: {encoding_dim_1}, {encoding_dim_2}, {encoding_dim_3}")
     


# Encoder
input_layer = layers.Input(shape=(input_dim,))

# Encoder layers
encoder = layers.Dense(encoding_dim_1, activation='relu')(input_layer)
encoder = layers.Dense(encoding_dim_2, activation='relu')(encoder)
encoder = layers.Dense(encoding_dim_3, activation='relu')(encoder)

# Decoder layers
decoder = layers.Dense(encoding_dim_2, activation='relu')(encoder)
decoder = layers.Dense(encoding_dim_1, activation='relu')(decoder)
decoder = layers.Dense(input_dim, activation='sigmoid')(decoder)

# Complete autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the model
autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss='mse',
    metrics=['mae']
)

# Display model summary
print("Autoencoder Architecture:")
autoencoder.summary()


# Train the autoencoder on normal transactions only
history = autoencoder.fit(
    X_train, X_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

print("\nTraining completed!")


# Plot training and validation loss
plt.figure(figsize=(14, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE During Training')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



# Predict on test data
reconstructions = autoencoder.predict(X_test)

# Calculate reconstruction error (MSE)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

print("Reconstruction Error Statistics:")
print(f"Mean: {mse.mean():.6f}")
print(f"Std: {mse.std():.6f}")
print(f"Min: {mse.min():.6f}")
print(f"Max: {mse.max():.6f}")
     


# Separate reconstruction errors for normal and fraud transactions
error_normal = mse[y_test == 0]
error_fraud = mse[y_test == 1]

# Plot distribution of reconstruction errors
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(error_normal, bins=50, alpha=0.7, label='Normal', color='blue')
plt.hist(error_fraud, bins=50, alpha=0.7, label='Fraud', color='red')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(range(len(error_normal)), error_normal, alpha=0.5,
            s=3, label='Normal', color='blue')
plt.scatter(range(len(error_fraud)), error_fraud, alpha=0.5,
            s=10, label='Fraud', color='red')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error: Normal vs Fraud')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Normal Transaction Error - Mean: {error_normal.mean():.6f}, Std: {error_normal.std():.6f}")
print(f"Fraud Transaction Error - Mean: {error_fraud.mean():.6f}, Std: {error_fraud.std():.6f}")
     


# Calculate threshold based on normal transaction errors
threshold_percentile = 95
threshold = np.percentile(error_normal, threshold_percentile)

print(f"Threshold (at {threshold_percentile}th percentile): {threshold:.6f}")

# Alternative: Mean + 3*Std
threshold_std = error_normal.mean() + 3 * error_normal.std()
print(f"Alternative Threshold (Mean + 3*Std): {threshold_std:.6f}")

# Use the percentile-based threshold
print(f"\nUsing threshold: {threshold:.6f}")



# Predict anomalies: if reconstruction error > threshold, it's an anomaly
predictions = (mse > threshold).astype(int)

# Print classification results
print("Predictions Distribution:")
print(f"Normal (0): {np.sum(predictions == 0)}")
print(f"Anomaly (1): {np.sum(predictions == 1)}")


# Calculate performance metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("=" * 50)
print("MODEL PERFORMANCE METRICS")
print("=" * 50)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")
print("=" * 50)



# Create confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Print detailed confusion matrix metrics
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix Breakdown:")
print(f"True Negatives (TN):  {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP):  {tp}")



# Visualize reconstruction errors with threshold
plt.figure(figsize=(14, 6))

# Plot reconstruction errors
plt.scatter(range(len(mse)), mse, c=y_test, cmap='coolwarm',
            alpha=0.5, s=10, label='Test Data')
plt.axhline(y=threshold, color='green', linestyle='--',
            linewidth=2, label=f'Threshold = {threshold:.4f}')
plt.xlabel('Sample Index')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs Threshold')
plt.legend()
plt.colorbar(label='Class (0=Normal, 1=Fraud)')
plt.grid(True, alpha=0.3)
plt.show()


# Show some sample predictions
print("Sample Predictions (First 20 test samples):")
print("-" * 70)
print(f"{'Index':<8} {'Actual':<10} {'Predicted':<12} {'Error':<15} {'Result':<10}")
print("-" * 70)

for i in range(20):
    actual = "Normal" if y_test[i] == 0 else "Fraud"
    predicted = "Normal" if predictions[i] == 0 else "Fraud"
    result = "✓" if y_test[i] == predictions[i] else "✗"
    print(f"{i:<8} {actual:<10} {predicted:<12} {mse[i]:<15.6f} {result:<10}")
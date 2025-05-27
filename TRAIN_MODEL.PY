import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Flatten

# Carregar os novos dados combinados
X = np.load("landmarks_combined.npy")
y = np.load("labels_combined.npy")

# Normalizar os landmarks
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Criar o modelo CNN + LSTM melhorado
model = Sequential([
    Conv1D(64, kernel_size=5, activation="relu", input_shape=(X_train.shape[1], X_train.shape[2])),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(256, return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.4),

    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),

    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

# Compilar o modelo
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treinar o modelo
history = model.fit(
    X_train, y_train,
    epochs=120,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# Salvar o modelo atualizado
model.save("fall_detection_cnn_lstm_v3.h5")
print("Modelo atualizado treinado e salvo!")

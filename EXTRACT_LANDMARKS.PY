import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Caminho base para os vídeos
base_path = r"C:\Users\Rafael\Desktop\archive\dataset\dataset"

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

SEQ_LENGTH = 50  # Mais frames por sequência

def smooth_landmarks(landmarks_seq, window=5):
    """Aplica um filtro de média móvel para suavizar os landmarks."""
    smoothed_seq = np.convolve(landmarks_seq, np.ones(window)/window, mode='same')
    return smoothed_seq

def extract_landmark_sequence(video_path, start_frame, end_frame):
    """Extrai landmarks ao longo de 50 frames e normaliza os dados."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    landmarks_seq = []

    for _ in range(SEQ_LENGTH):
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            landmarks = np.array(landmarks).flatten()
            landmarks_seq.append(landmarks)
        else:
            landmarks_seq.append(np.zeros(33 * 3))

    cap.release()

    while len(landmarks_seq) < SEQ_LENGTH:
        landmarks_seq.append(np.zeros(33 * 3))

    # Aplicar suavização
    landmarks_seq = np.array(landmarks_seq)
    for i in range(landmarks_seq.shape[1]):  
        landmarks_seq[:, i] = smooth_landmarks(landmarks_seq[:, i])

    return landmarks_seq

# Criar estrutura para armazenar os dados
sequences = []
labels = []

# Ler o CSV com os intervalos de frames e labels
data_tuple = pd.read_csv("data_tuple3.csv")
data_tuple = data_tuple[(data_tuple["cam"] >= 1) & (data_tuple["cam"] <= 8)]

for _, row in data_tuple.iterrows():
    chute = int(row["chute"])
    cam = int(row["cam"])
    start_frame = int(row["start"])
    end_frame = int(row["end"])
    label = int(row["label"])
    
    video_path = os.path.join(base_path, f"chute{chute}cam{cam}.avi")
    
    if os.path.exists(video_path):
        sequence = extract_landmark_sequence(video_path, start_frame, end_frame)
        sequences.append(sequence)
        labels.append(label)
    else:
        print(f"Vídeo não encontrado: {video_path}")

# Converter para arrays numpy
X = np.array(sequences)
y = np.array(labels)

np.save("landmarks_sequences.npy", X)
np.save("labels.npy", y)

print(f"Dados processados e salvos! {X.shape} sequências extraídas.")


# Carregar os dois datasets
X_old = np.load("landmarks_sequences.npy")
y_old = np.load("labels.npy")

X_le2i = np.load("landmarks_le2i.npy")
y_le2i = np.load("labels_le2i.npy")

# Concatenar os dados
X_combined = np.concatenate((X_old, X_le2i), axis=0)
y_combined = np.concatenate((y_old, y_le2i), axis=0)

# Salvar o novo dataset combinado
np.save("landmarks_combined.npy", X_combined)
np.save("labels_combined.npy", y_combined)

print(f"Novo dataset combinado: {X_combined.shape} sequências.")


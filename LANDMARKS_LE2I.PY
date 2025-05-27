import os
import cv2
import mediapipe as mp
import numpy as np

# Caminho base do dataset LE2I
base_path = r"C:\Users\Rafael\Downloads\base le2i"

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

SEQ_LENGTH = 50  # Número de frames por sequência

def extract_landmarks_from_video(video_path, start_frame, end_frame):
    """Extrai landmarks de um vídeo nos frames relevantes."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f" Erro ao abrir o vídeo: {video_path}")
        return None  # Ignora vídeos que não podem ser abertos

    # Configurar o frame inicial
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    landmarks_seq = []

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            landmarks_seq.append(np.array(landmarks).flatten())
        else:
            landmarks_seq.append(np.zeros(33 * 3))  # Se não detectar landmarks, preencher com zeros

    cap.release()

    # Garantir que a sequência tenha SEQ_LENGTH frames
    while len(landmarks_seq) < SEQ_LENGTH:
        landmarks_seq.append(np.zeros(33 * 3))

    return np.array(landmarks_seq)

# Criar listas para armazenar dados
sequences = []
labels = []

# Pastas do dataset LE2I
locations = ["Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02"]

for location in locations:
    annotation_folder = os.path.join(base_path, location, location, "Annotation_files")
    video_folder = os.path.join(base_path, location, location, "Videos")

    # 📌 Verificar se a pasta existe antes de tentar acessar
    if not os.path.exists(annotation_folder):
        print(f" Pasta de anotações não encontrada: {annotation_folder}")
        continue

    if not os.path.exists(video_folder):
        print(f"Pasta de vídeos não encontrada: {video_folder}")
        continue

    # Processar cada arquivo de anotação
    for annotation_file in os.listdir(annotation_folder):
        if annotation_file.endswith(".txt"):
            video_id = annotation_file.replace("video (", "").replace(").txt", "")
            annotation_path = os.path.join(annotation_folder, annotation_file)
            video_path = os.path.join(video_folder, f"video ({video_id}).avi")

            # 📌 Verificar se o vídeo existe antes de tentar abrir
            if not os.path.exists(video_path):
                print(f" Vídeo não encontrado: {video_path}")
                continue

            # Ler o arquivo de anotação
            with open(annotation_path, "r") as file:
                lines = file.readlines()

            # Garantir que há pelo menos duas linhas no arquivo
            if len(lines) < 2:
                print(f" Arquivo incompleto: {annotation_file}")
                continue

            # Verificar se as primeiras linhas são numéricas
            try:
                start_frame = int(lines[0].strip())
                end_frame = int(lines[1].strip())
            except ValueError:
                print(f" Formato inesperado no arquivo: {annotation_file}")
                continue

            # Extrair landmarks da queda
            landmarks_seq = extract_landmarks_from_video(video_path, start_frame, end_frame)

            if landmarks_seq is not None:
                # Adicionar sequência e rótulo
                sequences.append(landmarks_seq)
                labels.append(1)  # 1 significa que é uma queda

# Salvar os dados extraídos
X_le2i = np.array(sequences)
y_le2i = np.array(labels)

np.save("landmarks_le2i.npy", X_le2i)
np.save("labels_le2i.npy", y_le2i)

print(f" Extração concluída: {X_le2i.shape} sequências de quedas salvas.")

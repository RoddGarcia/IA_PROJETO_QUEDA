import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Carregar o modelo treinado
model = load_model("fall_detection_cnn_lstm_v3.h5")

SEQ_LENGTH = 50  # O mesmo número de frames usado no treinamento
landmark_queue = deque(maxlen=SEQ_LENGTH)

def process_video(video_path, output_path):
    """Processa um vídeo e salva o resultado com landmarks e detecção de quedas."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    fall_detected = False
    status = "Nao Queda"
    fall_counter = 0  # 🔹 Agora inicializa corretamente antes do loop

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
            landmarks = np.array(landmarks).flatten()
            landmark_queue.append(landmarks)

            if len(landmark_queue) == SEQ_LENGTH:
                prediction = model.predict(np.expand_dims(np.array(landmark_queue), axis=0))[0][0]
                
                if prediction > 0.5:
                    fall_counter += 1  # 🔹 Agora só tenta incrementar se `fall_counter` existir
                else:
                    fall_counter = 0  # 🔹 Evita valores indefinidos

                if fall_counter >= 5:  # Só confirma queda se detectar 5 frames seguidos
                    status = "Queda"
                    fall_detected = True
                elif not fall_detected:
                    status = "Nao Queda"

        # Desenhar landmarks no vídeo
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Exibir o status no vídeo
        text_position = (int(width / 2) - 100, 50)
        color = (0, 0, 255) if status == "Queda" else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Processamento concluído. Vídeo salvo em: {output_path}")

# Testar o modelo em um vídeo de entrada
process_video(r"C:\Users\Rafael\Desktop\archive\dataset\dataset\chute7cam7.avi", "output_video.avi")

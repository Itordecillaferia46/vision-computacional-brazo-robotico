import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Configurar captura de video
cap = cv2.VideoCapture(0)

# Variables para seguimiento de estado de la mano
mano_abierta = False
mano_cerrada = False

while cap.isOpened():
    # Capturar frame
    ret, frame = cap.read()
    if not ret:
        continue

    # Convertir el frame a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el frame con MediaPipe Hands
    results = hands.process(frame_rgb)

    # Verificar si se detectaron manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Aquí puedes procesar los landmarks de la mano para detectar movimientos
            dedo_pulgar = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            dedo_indice = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            dedo_corazon = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            dedo_anular = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            dedo_menique = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Calcular distancia entre la punta del pulgar y la punta del índice
            distancia = abs(dedo_pulgar.x - dedo_indice.x) + abs(dedo_pulgar.y - dedo_indice.y)

            # Determinar si la mano está abierta o cerrada
            if distancia > 0.1:  # Ajusta este umbral según tu necesidad
                mano_abierta = True
                mano_cerrada = False
            else:
                mano_abierta = False
                mano_cerrada = True

    # Mostrar el estado de la mano
    if mano_abierta:
        cv2.putText(frame, 'Mano Abierta', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    elif mano_cerrada:
        cv2.putText(frame, 'Mano Cerrada', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # elif not results.multi_hand_landmarks:
    #     cv2.putText(frame, 'Sin detectar mano', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Mostrar el frame
    cv2.imshow('Hand Tracking', frame)

    # Salir si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()


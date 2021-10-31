from google.protobuf.json_format import MessageToDict
import cv2
import mediapipe as mp
import keras
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

labels = ["clear", "eraser", "nextpage", "pendown", "penup", "previouspage", "quickcolour1", "quickcolour2", "quickcolour3"]
model = keras.models.load_model('OptionRecognition.h5')

# For webcam input:
vid = cv2.VideoCapture(0)

while True:
  ret, image = vid.read()
  with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4) as hands:
    image = cv2.flip(image, 1)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    index = 0
    left = False
    if not results.multi_hand_landmarks:
      print("No hands visible.")
      continue
    for m in results.multi_handedness:
        handedness_dict = MessageToDict(m)
        if (handedness_dict["classification"][0]["label"]) == "Left":
          left = True
          break
        index += 1
    if(left != True):
        print("Left hand not shown.")
        continue

    # Only consider left hand (for a right handed person)

    row = []
    origin = results.multi_hand_landmarks[index].landmark[0]
    origin_x = origin.x
    origin_y = origin.y
    origin_z = origin.z
    for landmark in results.multi_hand_landmarks[index].landmark:
        # Get relative positions

        row.append(origin_x - landmark.x)
        row.append(origin_y - landmark.y)
        row.append(origin_z - landmark.z)
    X = [row]
    predictions = model.predict(X)
    print(labels[np.argmax(predictions[0])])
  if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
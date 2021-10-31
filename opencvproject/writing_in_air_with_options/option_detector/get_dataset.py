# Importing required libraries

import cv2
import math
import numpy as np
import os
import mediapipe as mp
import csv

# Write resultant data to gestures.csv file

fields = list(range(63))
fields.append("label")
filename = "gestures.csv"
rows = []
labels = ["clear", "eraser", "nextpage", "pendown", "penup", "previouspage", "quickcolour1", "quickcolour2", "quickcolour3"]

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow("", img)


for label in labels:
  print("./final_dataset/"+label+"/")
  path = os.walk("./final_dataset/"+label+"/")

  fileso =[]

  for root, directories, files in path:
    for directory in directories:
        print(directory)
    for file in files:
        print(file)
        fileso.append(file)

# Read images with OpenCV
  images = {name: cv2.imread("./final_dataset/"+label+"/"+name) for name in fileso}
# Preview the images
  for name, image in images.items():
    print(name)   
    resize_and_show(image)


  mp_hands = mp.solutions.hands
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles

# Run MediaPipe Hands
  from google.protobuf.json_format import MessageToDict
  with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.4) as hands:
    for name, image in images.items():
    # Convert the BGR image to RGB, flip the image around y-axis for correct 
    # handedness output and process it with MediaPipe Hands
      results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))

    # Print handedness (left v.s. right hand)
      print(f'Handedness of {name}:')
      print(results.multi_handedness)

      if not results.multi_hand_landmarks:
        continue
    # Draw hand landmarks of each hand
      print(f'Hand landmarks of {name}:')
      image_hight, image_width, _ = image.shape
      annotated_image = cv2.flip(image.copy(), 1)
      for hand_landmarks in results.multi_hand_landmarks:
      # Print index finger tip coordinates
        print(
            f'Index finger tip coordinate: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        )
        mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      resize_and_show(cv2.flip(annotated_image, 1))
      index = 0
      left = False
      for m in results.multi_handedness:
        handedness_dict = MessageToDict(m)
        if (handedness_dict["classification"][0]["label"]) == "Left":
          left = True
          break
        index += 1
      if(left != True):
        continue
      # Only consider left hand (for a right handed person)
      print("Left handed one is : ", index)
      row = []
      origin = results.multi_hand_landmarks[index].landmark[0]
      origin_x = origin.x
      origin_y = origin.y
      origin_z = origin.z
      for landmark in results.multi_hand_landmarks[index].landmark:
        # Get relative positions
        print(origin_x - landmark.x)
        print(origin_y - landmark.y)
        print(origin_z - landmark.z)
        row.append(origin_x - landmark.x)
        row.append(origin_y - landmark.y)
        row.append(origin_z - landmark.z)
      row.append(label)
      rows.append(row)
    

with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)
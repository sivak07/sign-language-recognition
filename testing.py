#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import tensorflow as tf


# In[5]:


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# In[ ]:


interpreter = tf.lite.Interpreter(model_path='model3.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
            continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:

            valu=results.multi_hand_landmarks
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                bountry=calc_bounding_rect(image,hand_landmarks)
                landmarks_=[]
                for idx in range(0,21):
                        landmarks_.append(hand_landmarks.landmark[idx].x)
                        landmarks_.append(hand_landmarks.landmark[idx].y)
                        landmarks_.append(hand_landmarks.landmark[idx].z)
                dict={0:'Right',1:'Left'}
                interpreter.set_tensor(input_details[0]['index'],np.array([landmarks_],dtype='float32'))
                interpreter.invoke()
                tflite_results = interpreter.get_tensor(output_details[0]['index'])
                info_text = handedness.classification[0].index
                cv2.rectangle(image,(bountry[0]-20,bountry[1]-20),(bountry[2]+20,bountry[3]+20),(255,128,0), 3)
                cv2.putText(image, f"{dict[info_text]}:{np.argmax(np.squeeze(tflite_results))+1}", (bountry[0]-20, bountry[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,128,0), 2)
                cv2.flip(image,1)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()


# In[8]:


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


# In[10]:


cap.release()
cv2.destroyAllWindows()


# In[ ]:





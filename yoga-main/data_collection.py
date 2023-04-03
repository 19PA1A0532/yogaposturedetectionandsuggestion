# import mediapipe as mp 
# import numpy as np 
# import cv2 
# import pandas as pd
# import xyz

# def inFrame(lst):
#     if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
#         return True 
#     return False

# # read image
# img = cv2.imread('yoga.png')

# name = input("Enter the name of the Asana : ")

# holistic = mp.solutions.pose
# holis = holistic.Pose()
# drawing = mp.solutions.drawing_utils

# X = []
# data_size = 0

# lst = []

# res = holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
#     # print("kdlfjkfjsld")
#     for i in res.pose_landmarks.landmark:
#         lst.append(i.x - res.pose_landmarks.landmark[0].x)
#         lst.append(i.y - res.pose_landmarks.landmark[0].y)

#     X.append(lst)
#     data_size = data_size+1

# np.save(f"{name}.npy", np.array(X))
# print("dflkfjsklsdks")
# print(np.array(X).shape)




# import mediapipe as mp 
# import numpy as np 
# import cv2 
# import pandas as pd
# import xyz

# def inFrame(lst):
# 	if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
# 		return True 
# 	return False
 
# cap = cv2.VideoCapture(0)

# name = input("Enter the name of the Asana : ")

# holistic = mp.solutions.pose
# holis = holistic.Pose()
# drawing = mp.solutions.drawing_utils

# X = []
# data_size = 0

# while True:
# 	lst = []

# 	_, frm = cap.read()

# 	frm = cv2.flip(frm, 1)

# 	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

# 	if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
# 		for i in res.pose_landmarks.landmark:
# 			lst.append(i.x - res.pose_landmarks.landmark[0].x)
# 			lst.append(i.y - res.pose_landmarks.landmark[0].y)

# 		X.append(lst)
# 		data_size = data_size+1

# 	else: 
# 		cv2.putText(frm, "Make Sure Full body visible", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

# 	drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

# 	cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

# 	cv2.imshow("window", frm)

# 	if cv2.waitKey(1) == 27 or data_size>80:
# 		cv2.destroyAllWindows()
# 		cap.release()
# 		break


# np.save(f"{name}.npy", np.array(X))
# print(np.array(X).shape)

import cv2
import numpy as np
import mediapipe as mp
image = cv2.imread('Triangle.jpg')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)
if results.pose_landmarks is not None:
    landmarks = np.array([[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark])
else:
    landmarks = np.zeros((33,2)) # if no landmarks detected, fill with zeros
np.save('triangle.npy', landmarks)
print(landmarks.shape)



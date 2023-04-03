import mediapipe as mp
import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from mpl_toolkits import mplot3d
from celluloid import Camera
from scipy import spatial
import pyshine as ps
import mongoconnect
from flask import Flask, render_template, Response , request,redirect, session
import cv2
from xyz import addFeatures, extractKeypoint,mp_pose,dif_compare,calculateAngle,compare_pose,diff_compare_angle,mp_drawing
import os
import urllib.request
app = Flask(__name__)
app.use_static_for_assets = True

import pymongo
myclient = pymongo.MongoClient("mongodb+srv://Pavitra:Pavistp6@cluster0.kwp8c5t.mongodb.net/?retryWrites=true&w=majority")
yogaposture = myclient["yoga_postures"]
users = yogaposture["users"]
app.secret_key = "mysecretkey"
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users.find_one({"username": username})

        if user and user["password"] == password:
            session["username"] = username
            return redirect("/")
        else:
            return "Invalid login credentials"

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        email = request.form["email"]

        user = users.find_one({"username": username})

        if user:
            return "Username already exists"

        new_user = {"username": username, "password": password, "email": email}
        users.insert_one(new_user)

        return redirect("/login")

    return render_template("signup.html")

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')



@app.route('/freetrail')
def get_freetrail():
    j = mongoconnect.k
    # print(j)
    return render_template('freetrail.html',url_list = j)


@app.route('/image')
def get_image():
    # Read the image file
    img = cv2.imread( addFeatures())
    _, img_encoded = cv2.imencode('.jpg', img)
    response = Response(img_encoded.tobytes(), mimetype='image/jpeg')
    return response

def generate_frames(path):
    cap = cv2.VideoCapture(0)  # Change this to the path of your video file if you want to display a pre-recorded video
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret,frame= cap.read()
            if not ret:
                break
            else:
                # Convert the image to RGB format and perform pose estimation
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ = image.shape
                image = cv2.resize(image, (int(image_width * (860 / image_height)), 860))

                try:
                    # path = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static\Boat.jpg'
                    x = extractKeypoint(path)
                    dim = (960, 760)
                    resized = cv2.resize(x[3], dim, interpolation = cv2.INTER_AREA)
                    # cv2.imshow('target',resized)
                    angle_target = x[2]
                    point_target = x[1]
                    landmarks = results.pose_landmarks.landmark
                    
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility*100, 2)]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility*100, 2)]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
                                round(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility*100, 2)]
                    
                    angle_point = []
                    
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    angle_point.append(right_elbow)
                    
                    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    angle_point.append(left_elbow)
                    
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    angle_point.append(right_shoulder)
                    
                    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    angle_point.append(left_shoulder)
                    
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                            
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    angle_point.append(right_hip)
                    
                    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    angle_point.append(left_hip)
                    
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    angle_point.append(right_knee)
                    
                    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angle_point.append(left_knee)
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                    
                    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    
                    
                    keypoints = []
                    for point in landmarks:
                        keypoints.append({
                            'X': point.x,
                            'Y': point.y,
                            'Z': point.z,
                            })
                    
                    p_score = dif_compare(keypoints, point_target)      
                    
                    angle = []
                    
                    angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
                    angle.append(int(angle1))
                    angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
                    angle.append(int(angle2))
                    angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
                    angle.append(int(angle3))
                    angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
                    angle.append(int(angle4))
                    angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
                    angle.append(int(angle5))
                    angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
                    angle.append(int(angle6))
                    angle7 = calculateAngle(right_hip, right_knee, right_ankle)
                    angle.append(int(angle7))
                    angle8 = calculateAngle(left_hip, left_knee, left_ankle)
                    angle.append(int(angle8))
                    
                    compare_pose(image, angle_point,angle, angle_target)
                    a_score = diff_compare_angle(angle,angle_target)
                    
                    if (p_score >= a_score):
                        cv2.putText(image, str(int((1 - a_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)

                    else:
                        cv2.putText(image, str(int((1 - p_score)*100)), (80,30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
        
                except:
                    pass


                # Draw pose landmarks on the image
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (0,0,255), thickness = 4, circle_radius = 4),
                                    mp_drawing.DrawingSpec(color = (0,255,0),thickness = 3, circle_radius = 3)
                                    )

                # Convert the image to JPEG format
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                # Yield the JPEG image as a bytes object
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/video_feed')
def video_feed():
    image_path = request.args.get('image_path')
    path = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static\{}'.format(image_path)
    print(path)
    return Response(generate_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(generate_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

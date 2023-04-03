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
from flask import Flask, render_template, Response , request,redirect, session, url_for, jsonify
import cv2
from xyz import addFeatures, extractKeypoint,mp_pose,dif_compare,calculateAngle,compare_pose,diff_compare_angle,mp_drawing
import os
import urllib.request
from datetime import datetime
from bson import ObjectId
# import camera
import pyttsx3

face_detection = mp.solutions.face_detection.FaceDetection()



app = Flask(__name__)
app.use_static_for_assets = True


import pymongo
myclient = pymongo.MongoClient("mongodb+srv://Pavitra:Pavistp6@cluster0.kwp8c5t.mongodb.net/?retryWrites=true&w=majority")
yogaposture = myclient["yoga_postures"]
users = yogaposture["users"]
collection = yogaposture['reminders']
app.secret_key = "mysecretkey"
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users.find_one({"username": username})

        if user and user["password"] == password:
            session["username"] = username
            reminduser = collection.find({'username': username})
            for i in reminduser:
                reminder = i['reminders']
                print(reminder)
        else:
            return "Invalid login credentials"
        
        return render_template('home.html', reminder=reminder)

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

        new_user = {"username": username, "password": password, "email": email, "poses_you_tried":[]}
        users.insert_one(new_user)
        collection.insert_one({'username':username,'reminders':[]})
        return redirect("/login")

    return render_template("signup.html")

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')


@app.route('/reminders', methods=['GET', 'POST'])
def reminders():
    if not session:
        return render_template('login.html')
    if request.method == 'POST':
        # Insert new reminder into database
        title = request.form['title']
        description = request.form['description']
        date = request.form['date']
        time = request.form['time']
        username = session['username']
        reminder = {'title': title, 'description': description, 'date': date, 'time': time, 'username': username}
        collection.update_one({'username': session["username"]}, {'$push': {'reminders': reminder}})
        return render_template('home.html')
    else:
        # Retrieve reminders from database
        username = session['username']
        reminders = collection.find({'username': username})
        for i in reminders:
            reminder = i['reminders']
        return render_template('reminder.html', reminders=reminder)



@app.route('/select')
def index1():
    if(session):
        return render_template('main.html')
    else:
        return render_template('login.html')
@app.route('/poses')
def poses():
    if(session):
        return render_template('poses.html')
    else:
        return render_template('login.html')
@app.route('/suggestedroutines')
def suggestedroutines():
    if(session):
        return render_template('suggestedroutines.html')
    else:
        return render_template('login.html')
@app.route('/other_page')
def other_page():
    j = mongoconnect.k
    return render_template('freetrail.html',url_list = j)

@app.route('/get_content')
def get_content():
    j = mongoconnect.k
    content = {'html': render_template('freetrail.html',url_list = j)}
    return jsonify(content)


@app.route('/select1', methods=['POST'])
def select():
    option = request.form['option']
    if option == 'Beginners':
        return redirect(url_for('page1'))
    elif option == 'Intermediate':
        return redirect(url_for('page2'))
    else:
        return redirect(url_for('page3'))

@app.route('/page1')
def page1():
    j = mongoconnect.beginners
    return render_template('freetrail.html',url_list = j,name = "Beginners Level Poses")

@app.route('/page2')
def page2():
    j = mongoconnect.intermediate
    return render_template('freetrail.html',url_list = j, name= "Intermediate Level Poses")

@app.route('/page3')
def page3():
    j = mongoconnect.experts
    return render_template('freetrail.html',url_list = j, name = "Experts Level Poses")

@app.route('/triedbyyou')
def get_triedByYou():
    u = users.find_one({"username":session["username"]})
    posesyoutried = u['poses_you_tried']
    return render_template('posesyoutried.html',url_list = posesyoutried)

@app.route('/')
def index():
    # print(session['username'])
    session.clear()
    return render_template('index.html')

@app.route('/home')
def home():
    # print(session['username'])
    if not session:
        return render_template('login.html')
    return render_template('home.html')

@app.route('/regular')
def get_regularposes():
    j = mongoconnect.k
    # print(j)
    return render_template('freetrail.html',url_list = j,name="Regular Poses")
@app.route('/weightloss')
def get_weightlossposes():
    j = mongoconnect.w
    # print(j)
    return render_template('freetrail.html',url_list = j,name ="Weight Loss")
@app.route('/fittness')
def get_fittnessposes():
    j = mongoconnect.f
    # print(j)
    return render_template('freetrail.html',url_list = j,name="Physical Fittness")
@app.route('/standing')
def get_standing():
    j = mongoconnect.standing
    # print(j)
    return render_template('pagefortypes.html',url_list = j,name="Standing")
@app.route('/prone')
def get_prone():
    j = mongoconnect.prone
    # print(j)
    return render_template('pagefortypes.html',url_list = j,name="Prone")
@app.route('/supine')
def get_supine():
    j = mongoconnect.supine
    # print(j)
    return render_template('pagefortypes.html',url_list = j,name="Supine")
@app.route('/seated')
def get_seated():
    j = mongoconnect.seated
    # print(j)
    return render_template('pagefortypes.html',url_list = j,name="Seated")
@app.route('/armlegsupport')
def get_armlegsupport():
    j = mongoconnect.armlegsupport
    # print(j)
    return render_template('pagefortypes.html',url_list = j,name="Arm And Leg Support")
@app.route('/armbalanceinversion')
def get_armbalanceinversion():
    j = mongoconnect.armbalanceinversion
    # print(j)
    return render_template('pagefortypes.html',url_list = j,name="Arm Balance Inversion")



@app.route('/begnningposes')
def get_beginingposes():
    j = mongoconnect.resbeginingposes
    # print(j)
    return render_template('pagefortypes1.html',url_list = j,name="Beginig Poses")
@app.route('/warmup')
def get_warmup():
    j = mongoconnect.reswarmup
    # print(j)
    return render_template('pagefortypes1.html',url_list = j,name="WarmUp Poses")
@app.route('/morningstretch')
def get_morningstretch():
    j = mongoconnect.resmorningstretch
    # print(j)
    return render_template('pagefortypes1.html',url_list = j,name="Morning Stretch")
@app.route('/restwithbalanceitout')
def get_restwithbalanceitout():
    j = mongoconnect.restwistbalanceitout
    # print(j)
    return render_template('pagefortypes1.html',url_list = j,name="Rest With Balance It Out")
@app.route('/nightpractice')
def get_nightpractice():
    j = mongoconnect.resnightpractice
    # print(j)
    return render_template('pagefortypes1.html',url_list = j,name="Night Practice")

@app.route('/freetrail')
def get_freetrail():
    j = mongoconnect.k
    # print(j)
    if(session):
        print(session)
        return render_template('freetrail.html',url_list = j,name="")
    return render_template('/')

@app.route('/pagefortypes')
def get_pagefortypes():
    j = mongoconnect.standing
    # print(j)
    if(session):
        print(session)
        return render_template('pagefortypes.html',url_list = j,name="")
    return render_template('/')

@app.route('/pagefortypes1')
def get_pagefortypes1():
    j = mongoconnect.standing
    # print(j)
    if(session):
        print(session)
        return render_template('pagefortypes1.html',url_list = j,name="")
    return render_template('/')


@app.route('/freetrail1')
def get_freetrail1():
    j = mongoconnect.k
    # print(j)
    if(session):
        print(session)
        return render_template('freetrail1.html',url_list = j,name="")
    return render_template('/')

@app.route('/image')
def get_image():
    # Read the image file
    img = cv2.imread( addFeatures())
    _, img_encoded = cv2.imencode('.jpg', img)
    response = Response(img_encoded.tobytes(), mimetype='image/jpeg')
    return response

def detailsofposture(path,secs):
    x = users.find({'username':session["username"]})
    poses = x["poses_you_tried"]
    for pose in poses:
        if(pose['img']==path):
            pose['time']= secs
    users.update_one({'username':session['username']},{'$set':{'poses_you_tried':poses}})
seconds = 0    
def generate_frames(path,posename):
    cap = cv2.VideoCapture(0)  # Change this to the path of your video file if you want to display a pre-recorded video
    starting_time = time.time()
    seconds = 0
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
                    seconds = int(time.time() - starting_time)
                    cv2.putText(image, posename+" seconds:"+str(int(time.time() - starting_time)), (400,40), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,255], 2, cv2.LINE_AA)
                    # print(int(time.time() - starting_time))
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color = (0,0,255), thickness = 4, circle_radius = 4),
                                    mp_drawing.DrawingSpec(color = (0,255,0),thickness = 3, circle_radius = 3)
                                    )
                else:
                    starting_time = time.time()
                    yoursecs = seconds
                    # detailsofposture(image_path,seconds)
                # Convert the image to JPEG format
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                
                # Yield the JPEG image as a bytes object
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # detailsofposture(path,seconds)
print(seconds)

@app.route('/video_feed')
def video_feed():
    image_path = request.args.get('image_path')
    path = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static\{}'.format(image_path)
    print(path)
    if(session):   
        print(path)
        print(session["username"])
        x = users.find({'username':session["username"]})
        posedata = mongoconnect.fulldata
        name = image_path.split('.')
        category_name = ""
        english_name = ""
        sanskrit_name_adapted = ""
        sanskrit_name =""
        translation_name =""
        pose_description = ""
        pose_benefits = ""
        for i in posedata:
            if(name[0] == i['english_name']):
                category_name = i['category_name']
                english_name = i['english_name']
                sanskrit_name_adapted = i['sanskrit_name_adapted']
                sanskrit_name = i['sanskrit_name']
                translation_name = i['translation_name']
                pose_benefits = i['pose_benefits']
                pose_description = i['pose_description']
        users.update_one({'username': session["username"]}, {'$push': {'poses_you_tried': {'img':image_path,'time':datetime.today(),'category_name':category_name,'english_name':english_name,
                                                                                           'sanskrit_name_adopted':sanskrit_name_adapted,'sanskrit_name':sanskrit_name,'translation_name':translation_name,
                                                                                           'pose_description':pose_description,'pose_benefits':pose_benefits}}})
    return Response(generate_frames(path,english_name), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(generate_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/video_feedx')
def video_feedx():
    image_path = request.args.get('image_path')
    path = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static\yoga_poses\{}'.format(image_path)
    # print(path)
    datas = image_path.split('.')
    data = datas[0]
    print(data)
    if(session):
        print(path)
        print(session["username"])
        x = users.find({'username':session["username"]})
        newimage_path = 'yoga_poses/'+image_path
        posedata = mongoconnect.alldata
        name = image_path.split('.')
        category_name = ""
        english_name = ""
        sanskrit_name_adapted = ""
        sanskrit_name =""
        translation_name =""
        pose_description = ""
        pose_benefits = ""
        for i in posedata:
            if(name[0] == i['pose_name']):
                category_name = i['Category']
                english_name = i['pose_name']
                # sanskrit_name_adapted = i['sanskrit_name_adapted']
                sanskrit_name = i['sanskrit_name']
                # translation_name = i['translation_name']
                pose_benefits = i['Benefits']
                pose_description = i['Description']
        users.update_one({'username': session["username"]}, {'$push': {'poses_you_tried': {'img':newimage_path,'time':datetime.today(),'category_name':category_name,'english_name':english_name,
                                                                                           'sanskrit_name_adopted':sanskrit_name_adapted,'sanskrit_name':sanskrit_name,'translation_name':translation_name,
                                                                                           'pose_description':pose_description,'pose_benefits':pose_benefits}}})   
    return Response(generate_frames(path,english_name), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed')
# def video_feed():
#     image_path = request.args.get('image_path')
#     path = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static\{}'.format(image_path)
    # if(session):
    #     print(path)
    #     print(session["username"])
    #     x = users.find({'username':session["username"]})
    #     users.update_one({'username': session["username"]}, {'$push': {'poses_you_tried': path}})
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#     # return Response(generate_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')

import pandas as pd
import gensim
import numpy as np
import pickle

model = gensim.models.Word2Vec.load(r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\model.bin')

# Load the SVM model
model1 = pickle.load(open(r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\Yoga-Asana-Recommendation-Model-main\savemodel.sav', "rb"))

# Get the list of unique words in the vocabulary
words = list(model.wv.key_to_index.keys())
dict_of_word_embeddings = dict({})
for i in words:
    dict_of_word_embeddings[i] = model.wv[i]

# Load the yoga asanas dataset
df = pd.read_csv("test.csv")
testdata= []
# df = pd.read(r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\test.csv')
# for i in df:
#     testdata.append({'asana':i['Asana'],'benifit':i['Benefits']})
asan = list(df['Asana'])
ben = list(df['Benefits'])
for i in range(len(asan)):
    asan[i] = asan[i].strip()
    testdata.append({'Asana':asan[i],'Benefits':ben[i]})
# print(testdata)
asanas = list(df['Asana'])
asana = []
for x in asanas:
  if x not in asana:
    asana.append(x)
from collections import Counter
from IPython.display import clear_output
def magic(user_input_words):
    predicted_asanas = []
    final_predicted_asanas = []
    for i in user_input_words:
        if i in dict_of_word_embeddings:
            input_array = np.expand_dims(dict_of_word_embeddings[i], axis=0)
            prediction = model1.predict(input_array)
            flatten_pred = prediction.flatten()
            result_indices = flatten_pred.argsort()[-10:][::-1]
            for result in result_indices:
                predicted_asanas.append(asana[result])
    counter_found = Counter(predicted_asanas)
    final_predicted_asanas_with_freq = counter_found.most_common(7)
  # # print(final_predicted_asanas_with_freq)

    for yoga, freq in final_predicted_asanas_with_freq:
        final_predicted_asanas.append(yoga)
  
    return final_predicted_asanas


@app.route('/recomendations', methods=['GET', 'POST'])
def get_recomendations():
    if(session):
        if request.method == 'POST':
            # Get the data from the form
            input1 = request.form['input1']
            input2 = request.form['input2']
            l = input1.split()
            l.extend(input2.split())
            for i in l:
                i = i.lower();
            print("hello")
            d = magic(l)
            nd = []
            for i in d:
                i = i.strip()
                nd.append(i)
            newnd = []
            for i in nd:
                for ind in testdata:
                    if(ind['Asana'] == i):
                        newnd.append({'english_name':ind['Asana'],'Benefits':ind['Benefits']})


            # print(newnd)
            return render_template('freetrail1.html',url_list = newnd ,name="Practice these ASANAS to overcome your problems")

        return render_template('recomendataion.html')
    return render_template('login.html')


# df= pd.read_csv('test.csv')

@app.route('/video_feedy')
def video_feedy():
    image_path = request.args.get('image_path')
    path = r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\static\images\images{}_3.jpg'.format(image_path)
    print(path)
    data = 'images/images'+image_path+'_3.jpg'
    # print(data)
    if(session):
        print(path)
        print(session["username"])
        x = users.find({'username':session["username"]})
        posedata =testdata
        name = image_path
        category_name = ""
        english_name = ""
        sanskrit_name_adapted = ""
        sanskrit_name =""
        translation_name =""
        pose_description = ""
        pose_benefits = ""
        for i in posedata:
            if(name == i['Asana']):
                # category_name = i['Category']
                english_name = i['Asana']
                # sanskrit_name_adapted = i['sanskrit_name_adapted']
                # sanskrit_name = i['sanskrit_name']
                # translation_name = i['translation_name']
                pose_benefits = i['Benefits']
                # pose_description = i['Description']
        users.update_one({'username': session["username"]}, {'$push': {'poses_you_tried': {'img':data,'time':datetime.today(),'category_name':category_name,'english_name':english_name,
                                                                                           'sanskrit_name_adopted':sanskrit_name_adapted,'sanskrit_name':sanskrit_name,'translation_name':translation_name,
                                                                                           'pose_description':pose_description,'pose_benefits':pose_benefits}}})   
    return Response(generate_frames(path,name), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')

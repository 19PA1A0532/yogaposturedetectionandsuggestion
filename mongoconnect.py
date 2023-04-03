weightLossPoses=["Plank","Warrior One","Triangle","Downward Dog","Shoulder Stand",
                 "Bridge","Chair","Bow","Sun"]

fitnessPoses=["Boat","Plank","Dolphin","Extended Side Angle","Locust","Plank","Side Angle","Triangle","Bow","Dog","Warriror One","Warrior Two","Warrior Three"]

regularPoses=["Child","Downward-Facing Dog","Plank","Four-Limbed Staff","Cobra","Tree","Triangle","Seated Half-Spinal Twist","Bridge","Corpse"]

import pymongo
import os
import urllib.request
myclient = pymongo.MongoClient("mongodb+srv://Pavitra:Pavistp6@cluster0.kwp8c5t.mongodb.net/?retryWrites=true&w=majority")
yogaposture = myclient["yoga_postures"]
yoga_category = yogaposture["yoga_category"]
yoga_poses_level = yogaposture["yoga_poses_level"]
yogadata = yogaposture["yogaData"]
doc = yoga_category.find()
doc1 = yoga_category.find()
doc2 = yoga_category.find()
doc3 = yoga_category.find()
doc4  = yoga_category.find()
fulldata = []
for i in doc4:
    x = i["poses"]
    for j in x:
        fulldata.append(j)
k = []
f = []
w = [] 

images=[]
poses=[]
def isink(x,k):
    bool = True
    for i in k:
        if(x == i["english_name"]):
            bool = False
            return bool
    return bool

beginners = []
intermediate = []
experts = []

document1 = yoga_poses_level.find({"difficulty_level":"Beginner"})
document2 = yoga_poses_level.find({"difficulty_level":"Intermediate"})
document3 = yoga_poses_level.find({"difficulty_level":"Expert"})
# print(document1)
def get_level(document):
    b = []
    # x = document['poses']
    for j in document:
        x = j['poses']
        for i in x:
            if(isink(i['english_name'],b)):
                b.append(i)
            else:
                continue
    return b
beginners = get_level(document1)
intermediate = get_level(document2)
experts = get_level(document3)
# print(beginners)
# print(intermediate)
# print(experts)
def get_pose_types(types,doc):
    f = []
    for i in doc:
        x = i["poses"]
        for j in x:
            if(j["english_name"] in types):
                if(isink(j["english_name"],f)):
                    f.append(j)
                    images.append(j["url_png"])
                    poses.append(j["english_name"])
                else:
                    continue
    return f
k = get_pose_types(regularPoses,doc1)
f = get_pose_types(fitnessPoses,doc2)
w = get_pose_types(weightLossPoses,doc3)
# kfwdata = k
# kfwdata.extend(f)
# kfwdata.extend(w)
# print(k)
# print(f)
# print(w)

def get_poses_data(name):
    res = [] 
    x = yogadata.find()
    for i in x:
        for j in i[name]:
            res.append(j)
    return res
alldata = []        
standing = get_poses_data("Standing")
prone = get_poses_data("Prone")
seated  = get_poses_data("Seated")
supine = get_poses_data("Supine")
armlegsupport = get_poses_data("Arm & Leg Support")
armbalanceinversion=get_poses_data("Arm Balance & Inversion")
alldata.extend(standing)
alldata.extend(prone)
alldata.extend(seated)
alldata.extend(supine)
alldata.extend(armlegsupport)
alldata.extend(armbalanceinversion)
def get_routines(routine):
    resroutine = []
    for j in routine:
        for i in alldata:
            if(j==i['pose_name']):
                resroutine.append(i)
    return resroutine
import requests
import os
def downloadIgmages(url,file_path,file_name):
    full_path = file_path + file_name + '.jpg'
    urllib.request.urlretrieve(url,full_path)
# for i in armbalanceinversion:
#     downloadIgmages(i['url_img'],'static/yoga_poses/',i['pose_name'])
# list_poses = [ ]
# for i in standing:
#     list_poses.append(i['pose_name'])
# for i in seated:
#     list_poses.append(i['pose_name'])
# for i in prone:
#     list_poses.append(i['pose_name'])
# for i in armlegsupport:
#     list_poses.append(i['pose_name'])
# for i in armbalanceinversion:
#     list_poses.append(i['pose_name'])
# for i in supine:
#     list_poses.append(i['pose_name'])
# print(list_poses)


morningstretch = ["Child's","Box","Cat","Cow","Cat","Cow","Cat","Box","Revolved Child's",
                  "Box","Revolved Child's","Box","Downward-Facing Dog","Lizard with Straight Arms",
                  "Lunge","Lunge on the Knee","Half Splits","Wide Splits","Seated Spinal Twist","Staff",
                  "Bridge","Wind Removing","Lizard with Straight Arms","Lunge",
                  "Lunge on the Knee","Standing Forward Bend","Chair","Mountain with Prayer Hands","Corpse","Banana","Corpse"]
beginningposes = ["Child's","Box","Cow","Cat","Box"]
warmup = ["Downward-Facing Dog","Standing Forward Bend","Halfway Lift","Standing Forward Bend","Mountain with Arms Up","Crescent Moon","Mountain with Arms Up","Mountain with Arms Up and Backbend","Crescent Moon","Mountain with Arms Up","Standing Forward Bend","Halfway Lift","Lunge","Plank","Low Push-up","Upward-Facing Dog"]
twistandbalanceitout = ["Downward-Facing Dog","Three Legged Downward-Facing Dog","Downward-Facing Dog with Knee to Forehead","Lunge on the Knee","Crescent Lunge on the Knee","Crescent Lunge Twist on the Knee","Crescent Lunge on the Knee","Lunge on the Knee","Lunge",
                       "Plank","Low Push-up","Upward-Facing Dog","Downward-Facing Dog","Three Legged Downward-Facing Dog","Three Legged Downward-Facing Dog with Hip Opener","Plank with Knee to Tricep","Flying Man","Three Legged Downward-Facing Dog","Lunge","Crescent Lunge","Crescent Lunge Halfway Fold","Warrior III",
                         "Crescent Lunge","Crescent Lunge Twist","Lunge","Plank","Low Push-up","Upward-Facing Dog","Downward-Facing Dog"]
hipopeningyin = []
bendmebindme = []
nightpractice = ["Child's","Box","Revolved Child's","Box","Revolved Child's","Box","Cow","Cat","Box","Downward-Facing Dog","Downward Facing Dog With Toe Raises","Rag Doll","Standing Forward Bend Twist","Halfway Lift","Awkward","Mountain","Mountain with Open Arm Twist","Mountain with Prayer Hands","Start with Hands Interlaced"
                 ,"Mountain","Garland","Extended Child's","Downward-Facing Dog","Low Cobra","Thunderbolt","Rabbit","Seated Spinal Twist","Head to Knee I","Half Locust","Extended Corpse","Corpse","Low Boat","Boat","Corpas","Happy Baby","Corpse","Bridge","Corpose","One Legged Wind Removing","Supine Spinal Twist","Supine Bound Angle","Corpose"]
abendroutine = []
mondaynightprivatesession = []
powercore = []

resmorningstretch  = (get_routines(morningstretch))
resbeginingposes= (get_routines(beginningposes))
reswarmup = (get_routines(warmup))
restwistbalanceitout = (get_routines(twistandbalanceitout))
resnightpractice = (get_routines(nightpractice))

# print(reswarmup)


def getDataofposecategory(name,doc,doc1):
    res=[]
    for i in doc:
        x = i["poses"]
        for j in x:
            if(j[0]==name):
                res.append(j)
    for i in doc1:
        y = i['difficulty_level']
        for j in y:
            if(j["english_name"]== name):
                res.append(j)
                break
    return res
def getDataofposesbyposes(name,alldata):
    res = []
    for i in alldata:
        if(i['pose_name']==name):
            res.append(i)
            break
    return res
docx = yoga_category.find()
documentx = yoga_poses_level.find()
# print(getDataofposecategory('Big Toe',docx,documentx))

import numpy as np
import pandas as pd
# testdata= []
# df = pd.read(r'D:\VIT[college work]\4-2mainproject\yogaposturedetection\test.csv')
# for i in df:
#     testdata.append({'asana':i['Asana'],'benifit':i['Benefits']})

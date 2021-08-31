from typing import ClassVar
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

path="img"
images=[]
classNames=[]

mylist=os.listdir(path)

# print(mylist)

for cl in mylist:
    curimg=cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classNames.append(os.path.splitext(cl)[0])

# print(classNames)
# print(images)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
        encodeone = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeone)
    return encodeList

def markAttendance(name):
    with open('attendance.csv','r+') as f:
        mydatalist=f.readlines()
        namelist=[]
        for line in mydatalist:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            date=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date}')
            




listof_encode=findEncodings(images)
# print(len(listof_encode[0]))

cap=cv2.VideoCapture(0)

while True:
    success, img= cap.read()
    # imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
    facecur=face_recognition.face_locations(imgS)
    encodecur=face_recognition.face_encodings(imgS,facecur)

    for encodeFace,faceLoc in zip(encodecur,facecur):
        match=face_recognition.compare_faces(listof_encode,encodeFace)
        facedis=face_recognition.face_distance(listof_encode,encodeFace)
        # print(facedis)
        matchIndex = np.argmin(facedis)
        # print(match)
        if match[matchIndex]:
            name=classNames[matchIndex].upper()
           

            cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)
            cv2.rectangle(img,(faceLoc[3],faceLoc[2]-35),(faceLoc[1],faceLoc[2]),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(faceLoc[3]+6,faceLoc[2]-6),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),2)
           
            markAttendance(name)

    cv2.imshow('Webcam',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if cv2.waitKey(0) == 13:
        break

# cap.release()
cv2.destroyAllWindows()






















 # y1,x2,y2,x1 = faceLoc
            # y1, x2, y2, x1 = y1,x2*1,y2*1,x1
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            # cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            # cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)



import cv2
import numpy as np
import face_recognition
import os
import itertools
from operator import itemgetter


#face_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')


path = 'img'
images = []
classNames = []
charList = os.listdir(path)
print(charList)
id = 0.9

for cl in charList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []
    for cimg in images:
        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(cimg)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)

# Library'de bulunan image'lerin encode sınırı

# Yüklenen resmin encode edilme bölgesi

resim = cv2.imread("test.jpg")
#imgS = cv2.resize(resim, (0, 0), None, 0.25, 0.25)
imgS = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)
encodeCurFrame = face_recognition.face_encodings(imgS)
#cv2.imshow("image deneme ", imgS)
#cv2.waitKey(0)
print("encodecurrent", encodeCurFrame)
# asfkasfalsf
print('Encoding Complated')
#print('deneme enc', encodeListKnown)
benzerlik = []

for item in encodeListKnown:
    print("item", encodeCurFrame)
    faceDis = face_recognition.face_distance(item, encodeCurFrame)
    matches = face_recognition.compare_faces(item, encodeCurFrame)
    if faceDis < id:
        matches = True
        benzerlik.extend(faceDis)
    else:
        print('Resim benzerlik göstermedi')

print("En benzer resim: ", min(benzerlik))

result = min(enumerate(benzerlik), key=itemgetter(1))[0]
print(result)


cv2.imshow('Captured Image', images[result])
cv2.waitKey(0)


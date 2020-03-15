import cv2
import numpy as np
import xlwrite,firebase.firebase_ini as fire;
import time
import sys
from playsound import playsound
start=time.time()
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read('trainer/trainer.yml')
id=0;
flag=0;
filename='filename';
dict ={
	'item1':1
}
font=cv2.FONT_HERSHEY_SIMPLEX
while(True):
	ret,img=cam.read();
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5);
	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		id,conf=rec.predict(gray[y:y+h,x:x+w])
		cv2.putText(img, str(id)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
	cv2.imshow("Face",img);
	if(cv2.waitKey(1)==ord('q')):
		break;
cam.release()
cv2.destroyAllWindows()

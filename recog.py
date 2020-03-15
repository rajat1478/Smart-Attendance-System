import cv2
import numpy as np
#import xlsxwriter
import xlwrite
import firebase.firebase_ini as fire;
import time
import sys

start=time.time()
period=8

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.load('training/training.yml')
id=0
font=cv2.FONT_HERSHEY_SIMPLEX
while(True):
	ret,img=cam.read();
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5);
	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		id,conf=rec.predict(gray[y:y+h,x:x+w])
		if(conf < 50):
			if(id==1):
				id='Shubam Walia'
				if((str(id)) not in dict):
					filename=xlwrite.output('attendance','class1',1,id,'yes');
					dict[str(id)]=str(id);

			elif(id==2):
				id = 'Rohan'
				if ((str(id)) not in dict):
					filename =xlwrite.output('attendance', 'class1', 2, id, 'yes');
					dict[str(id)] = str(id);

			elif(id==3):
				id = 'Raveen'
				if ((str(id)) not in dict):
					filename =xlwrite.output('attendance', 'class1', 3, id, 'yes');
					ict[str(id)] = str(id);

			elif(id==4):
				id = 'Sonu'
				if ((str(id)) not in dict):
					filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
					dict[str(id)] = str(id);

			else:
				id = 'Unknown, can not recognize'
				break

		cv2.putText(img, str(id)+" "+str(conf),(x,y-10),font,0.55,(120,255,120),1)
	cv2.imshow("Face",img);
	if(cv2.waitKey(1)==ord('q')):
		break;
cam.release()
cv2.destroyAllWindows()


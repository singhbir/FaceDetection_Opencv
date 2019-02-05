import cv2
import numpy as np

face_detect=cv2.CascadeClassifier('haar_frontal.xml')  #haar and LBP
eyes_detect=cv2.CascadeClassifier('haar_eyes.xml')
cam=cv2.VideoCapture(0)

# while (True):
#     ret,img=cam.read()
#     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     faces=face_detect.detectMultiScale(gray,1.3,5)
#     eyes=eyes_detect.detectMultiScale(gray,1.3,5)
#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#     for (x,y,w,h) in eyes:
#         cv2.rectangle(img,(x,y),(x+h,y+h),(0,0,255),2)
#     cv2.imshow("faces",img)
#     if cv2.waitKey(0)==ord("q"):
#         break
# cam.release()
# cv2.destroyAllWindows()

while(cam.isOpened()):
    ret,img = cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detect.detectMultiScale(gray,1.3,5)
    eyes=eyes_detect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        for (x,y,w,h) in eyes:
            cv2.rectangle(img,(x,y),(x+h,y+h),(0,0,255),2)
    cv2.imshow("faces",img)
    if cv2.waitKey(35)==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()

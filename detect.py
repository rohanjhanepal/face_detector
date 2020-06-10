import cv2
from PIL import Image


detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')
vid = cv2.VideoCapture(0)

while True:
    rect , frame = vid.read()

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    det = detector.detectMultiScale(gray, 1.3,5)
    copied = frame.copy()
    fr= frame
    for (x,y,w,z) in det:
##        a = int((x+2.5*w)/2)
##        b= int((y+2*z)/2)
        
       
        #cv2.circle(frame,(a,b), 100,(73,7,39),3)
        cv2.rectangle(frame , (x,y) , (x+w,y+z) , (0,102,255) , -1)
       
        
                
       
        #print(x,y)
    alpha = 0.4  

   
    image_new = cv2.addWeighted(copied, alpha, frame, 1 - alpha, 0)
    
    cv2.imshow('win' ,fr)

    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()


    

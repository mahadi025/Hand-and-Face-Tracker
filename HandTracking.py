import cv2 as cv
import mediapipe as mp
import time


cap = cv.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
x1 = 0
y1 = 0
x2 = 0
y2 = 0
x3 = 0
x4 = 0
x5 = 0
x6 = 0
while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                if id == 0:
                    x1 = cx
                    y1 = cy
                if id == 2:
                    cv.circle(img, (cx, cy), 5, (255, 255, 0), cv.FILLED)
                if id == 5:
                    x2 = cx
                    y2 = cy
                    # cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
                if id == 9:
                    x3 = cx
                    y3 = cy
                    # cv.circle(img, (cx, cy), 10, (255, 255, 0), cv.FILLED)
                if id == 13:
                    x4 = cx
                    y4 = cy
                if id == 17:
                    x5 = cx
                    y5 = cy
                if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                    X = int((x1+x2)/2)
                    Y = int((y1+y2)/2)
                    cv.circle(img, (X, Y), 5, (0, 0, 0), cv.FILLED)
                if x3 != 0 and y3 != 0 and x2 != 0 and y2 != 0:
                    X = int((x3+x2)/2)
                    Y = int((y3+y2)/2)
                    cv.circle(img, (X, Y), 5, (0, 0, 255), cv.FILLED)
                if x3 != 0 and y3 != 0 and x4 != 0 and y4 != 0:
                    X = int((x3+x4)/2)
                    Y = int((y3+y4)/2)
                    cv.circle(img, (X, Y), 5, (0, 255, 255), cv.FILLED)
                if x5 != 0 and y5 != 0 and x4 != 0 and y4 != 0:
                    X = int((x5+x4)/2)
                    Y = int((y5+y4)/2)
                    cv.circle(img, (X, Y), 5, (255, 0, 255), cv.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    # Face Detection
    gray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    haar_cascade=cv.CascadeClassifier('haar_face.xml')
    faces_rect=haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x,y,w,h) in faces_rect:
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    img = cv.flip(img, 1)
    cv.putText(img, str(int(fps)), (10, 70),
               cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
    cv.imshow('Webcam', img)
    if cv.waitKey(10) & 0xff == ord('d'):
        break

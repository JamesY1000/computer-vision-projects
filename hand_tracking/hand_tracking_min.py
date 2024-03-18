import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # Adjust the index based on the warning message
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Functions from mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0


while True:
    
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Hands uses only RGB images
    results = hands.process(imgRGB)
    # Print(results.multi_hand_landmarks) #prints the pose of hand landmarks
    
    # Draws hand points and connections for each hand
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # For each pose hand

            for id, lm in enumerate(handLms.landmark):
                print(id, lm) # Prints ratio of image for landmark position

                # Converts ratio to pixel coordinates
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                #Draws circle on landmark 1 (palm)
                if id == 0:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # Draws hand points/connections

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    #Display fps counter
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3,
                (255, 0, 255), 3)



    cv2.imshow("Image", img)
    cv2.waitKey(1)



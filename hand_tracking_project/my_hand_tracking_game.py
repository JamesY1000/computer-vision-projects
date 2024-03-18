import cv2
import time
import mediapipe as mp
import hand_tracking_module as htm


prevTime = 0
currTime = 0

# Initialise camera image
cap = cv2.VideoCapture(0)  # Adjust the index based on the warning message
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

detector = htm.handDetector()

while True:
    success, img = cap.read()
    hands_found = detector.findHands(img, draw=True)
    # Returns list of landmark positions
    lmlist = detector.findPosition(hands_found, draw=False)
    
    # Print positions of thumb (landmark 4)
    
    if len(lmlist) !=0:
        print(lmlist[4])


    # Create fps counter
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, "FPS counter: " + str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 255, 0), 1)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
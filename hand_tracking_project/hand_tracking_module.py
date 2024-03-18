import cv2
import mediapipe as mp
import time


class handDetector():
    # Initialise variables
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Creates object, whose variable is static_image_mode. Now we call the variable of the object self.static_image_mode = static_image_mode
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_cofidence = min_tracking_confidence

        
        # Functions from mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode, self.max_num_hands, self.model_complexity, self.min_detection_confidence, self.min_tracking_cofidence)
        self.mpDraw = mp.solutions.drawing_utils

    # If you have self as a parameter, you can call all instances of the object self and use its variables eg. self.hands
    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Hands uses only RGB images
        self.results = self.hands.process(imgRGB)
        # Print(results.multi_hand_landmarks) #prints the pose of hand landmarks
        
        # Draws hand points and connections for each hand
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks: # For each pose hand
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # Draws hand points/connections

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmlist = []

        if self.results.multi_hand_landmarks:
                self.myHand = self.results.multi_hand_landmarks[handNo]

                for id, lm in enumerate(self.myHand.landmark):
                    print(id, lm) # Prints ratio of image for landmark position
                    # Converts ratio to pixel coordinates
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    lmlist.append([id, cx, cy])

                    
                    #Draws circle on landmark 1 (palm)
                    if draw:
                        if id == 0:
                            cv2.circle(img, (cx, cy), 25, (0, 255, 0), cv2.FILLED)

        return lmlist


def main():
    prevTime = 0
    currTime = 0

    # Initialise camera image
    cap = cv2.VideoCapture(0)  # Adjust the index based on the warning message
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    detector = handDetector()

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
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ =="__main__":
    main()


import cv2
import mediapipe as mp
import pose_module as pm
import time

# Load video into cap
video_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/PoseVideos/"
cap = cv2.VideoCapture(video_path + "video6.mp4")

prev_time = 0
detector = pm.poseDetector(model_complexity=1)

while True:
    success, img = cap.read()
    detector.findPose(img, draw=True)
    lmlist = detector.findPosition(img, draw=True)

    # Print values of selected landmarks, then draw our own circles on them
    print("lm [%s]: %s \nlm [%s]: %s \nlm [%s]: %s" %(25, lmlist[25], 15, lmlist[15], 5, lmlist[5]))
    cv2.circle(img, (lmlist[25][1], lmlist[25][2]), 10, (150, 50, 100), cv2.FILLED)
    cv2.circle(img, (lmlist[15][1], lmlist[15][2]), 10, (100, 100, 255), cv2.FILLED)
    cv2.circle(img, (lmlist[5][1], lmlist[5][2]), 10, (50, 255, 0), cv2.FILLED)

    # Display FPS
    curr_time = time.time()
    fps = 1/ (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, "FPS Counter: " +str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    # Display video with our modifications
    cv2.imshow("Random video lol", img)
    # Waits for 1 milisecond after video finishes playing before destroying video window
    cv2.waitKey(1)


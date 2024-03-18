import cv2 
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()

video_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/PoseVideos/"
cap = cv2.VideoCapture(video_path + "video2.mp4")

prev_time = 0


while True:
    success, img = cap.read()
    cv2.imshow("Random video lol", img)


    # Display FPS
    curr_time = time.time()
    fps = 1/curr_time - prev_time
    prev_time = curr_time
    cv2.putText(img, "FPS counter: " + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0, 0, 255), 3)



    cv2.waitKey(1)

    
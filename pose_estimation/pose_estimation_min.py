import cv2 
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

# Load video into cap
video_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/PoseVideos/"
cap = cv2.VideoCapture(video_path + "video4.mp4")

prev_time = 0


while True:
    success, img = cap.read()

    # Read colour in RGB - image convert from BGR --> RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Processing and displaying pose landmark locations
    result = pose.process(imgRGB)
    print(result.pose_landmarks)

    # Draw pose lines
    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # For each set of pose landmarks, print landmark information (x, y)
        for id, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, lm)

            # Convert image ratio value to pixel value
            x_pixel_value = int(lm.x * w)
            y_pixel_value = int(lm.y * h)

            # Overlay our circle onto detected landmarks
            cv2.circle(img, (x_pixel_value, y_pixel_value), 10, (0, 100, 255), cv2.FILLED)



    # Display FPS
    curr_time = time.time()
    fps = 1/ (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(img, "FPS Counter: " +str(int(fps)), (70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    # Display video with our modifications
    cv2.imshow("Random video lol", img)


    # Waits for 1 milisecond after video finishes playing before destroying video window
    cv2.waitKey(1)

    
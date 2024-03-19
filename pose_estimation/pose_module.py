import cv2 
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


class poseDetector():
    # Initialises all the variables of the objects for use in all methods throughout the class
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False, 
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mpDraw
        self.mpPose = mpPose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.enable_segmentation,
                                     self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)
    
    # Draws circles and connecting lines for landmarks 
    def findPose(self, img, draw=True):

    # Read colour in RGB - image convert from BGR --> RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Processing and displaying pose landmark locations
        self.result = self.pose.process(imgRGB)
        # print(result.pose_landmarks)

        # Draw pose lines
        if self.result.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.result.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    
    # Returns pixel coordinates of landmarks drawn
    def findPosition(self, img, draw=True):
        lmlist = []
        if self.result.pose_landmarks:
            # For each set of pose landmarks, print landmark information (x, y)
            for id, lm in enumerate(self.result.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)

                # Convert image ratio value to pixel value
                x_pixel_value = int(lm.x * w)
                y_pixel_value = int(lm.y * h)
                lmlist.append([id, x_pixel_value, y_pixel_value])

                # Overlay our circle onto detected landmarks
                # cv2.circle(img, (x-pixel_value, y_pixel_value), 10, (150, 50, 100), cv2.FILLED)
  

        return lmlist



def main():

    # Load video into cap
    video_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/PoseVideos/"
    cap = cv2.VideoCapture(video_path + "video1.mp4")

    prev_time = 0
    detector = poseDetector(model_complexity=1)

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

    
if __name__ == "__main__":
    main()

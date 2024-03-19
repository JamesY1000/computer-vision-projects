import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()

class faceDetector():
    # Initialise objects and their variables to be used throughout the class
    def __init__(self, min_detection_confidence=0.5):
        self.min_detection_confidence = min_detection_confidence

        self.mpDraw = mpDraw
        self.mpFaceDetection = mpFaceDetection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence)


    # Draw landmarks and connecting points
    def findFace(self, img, draw=True):
        # Detect faces, draw circles and store landmark information
        
        # Convert image from BGR --> RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process landmark information
        self.result = self.faceDetection.process(imgRGB)
        print(self.result.detections)

        bbox_coords = []

        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                # Display each detection information separately
                
                # Obtain bbox information - xmin, ymin, width, height, x, y
                bbox_relative = detection.location_data.relative_bounding_box

                # Obtain detection confidence
                det_conf = detection.score
                
                # Convert locations to pixels
                h, w, c = img.shape
                left_bound = int(bbox_relative.xmin * w)
                lower_bound = int(bbox_relative.ymin * h)
                right_bound = int(bbox_relative.width * w)
                upper_bound = int(bbox_relative.height * h) 

                bbox_coords.append([left_bound, lower_bound, right_bound, upper_bound])

                print(f"Left boundary: {bbox_coords[0][0]}", f"Lower boundary: {bbox_coords[0][1]}", f"Right boundary: {bbox_coords[0][2]}", f"Upper boundary: {bbox_coords[0][3]}")

                if draw:
                    cv2.rectangle(img, (left_bound, lower_bound, right_bound, upper_bound), (0, 255, 0), 1)

                    # Add detection confidence
                    cv2.putText(img, f"Det conf: {str(int(det_conf[0] * 100))}%", (left_bound, lower_bound - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 100, 255), 1)
            
        return img, bbox_coords




    

def main():
    
    video_path = "/mnt/c/Users/james/OneDrive/Desktop/Computer vision videos/FacialVideos/"
    cap = cv2.VideoCapture(video_path + "video1.mp4")
    prev_time = 0
    detector = faceDetector()

    while True:
        #Load videos
        success, img = cap.read()
        if not success:
            break

        img, bbox_coords = detector.findFace(img, draw=True)
        
        
        # Display fps
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(img, f'FPS counter: {str(int(fps))}', (50, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)


        # Show image
        cv2.imshow('Facial detection module', img)
        # Delays 1 ms between frames - higher the value, the more your video is slowed down (reducing FPS)
        cv2.waitKey(65)



if __name__ == "__main__":
    main()
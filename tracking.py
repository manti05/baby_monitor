import cv2
import mediapipe as mp
import math
import numpy as np
from yunet import YuNet
import time
import logging

logger = logging.getLogger(__name__)

class PoseEstimation():
    def __init__(self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = False,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.4):
        # False for Video feed
        self.mode = static_image_mode
        #
        self.comp = model_complexity
        # Smooth out the landmarks
        self.smooth = smooth_landmarks
        #
        self.enSeg = enable_segmentation
        #
        self.smoothSeg = smooth_segmentation
        # How sensitive the detection / tracking is (0.5)
        self.minDetect = min_detection_confidence
        self.minTrack = min_tracking_confidence

        # Initialising Pose Object with all variables / options needed for our use case
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.comp, self.smooth, self.enSeg, self.smoothSeg, self.minDetect, self.minTrack)

    # Useful for troubleshooting body tracking
    def drawLandmarks(self, image, draw=True):
        # Image is already being passed in RGB
        results = self.pose.process(image)
        # If Landmarks are detected
        if results.pose_landmarks:
            # if Draw flag is True (Draw Landmarks on Frame)
            if draw:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        # return trackingImage

    def babyPosition(self, image):
        baby_position = "Covered"
        image_height = 480
        image_width = 640
        output = image.copy()
        results = self.pose.process(output)
        # If body is not covered and can be tracked
        if results.pose_landmarks:
            lefteyeX = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x
            righteyeX = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x
            eyesPosition = righteyeX - lefteyeX
            # Tracking Z value of Baby's shoulders (left and right shoulders)
            leftShoulderZ = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z * image_height)
            rightShoulderZ = (results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z * image_width)
            # Comparing the difference in Z values and seeing how big difference between them is
            # If the 2 values are positive and negative (one positive and the other negative)
            # the baby is on its side as the difference between the Z values can't be that
            # large if the Baby is laying down flat (Face Up/ Face Down)
            # Therefore if the values are either both positive or both negative baby is not on its side
            if leftShoulderZ < 0 and rightShoulderZ > 0 or leftShoulderZ > 0 and rightShoulderZ < 0:
                differenceZ = -1 * (leftShoulderZ + rightShoulderZ)
                if differenceZ > 30 or differenceZ < -30:
                    baby_position = "On It's Side"
            elif leftShoulderZ > 0 and rightShoulderZ > 0 or leftShoulderZ < 0 and rightShoulderZ < 0:
                # as right eyes x value is greater than left eye x value
                # eyesposition will be positive
                # eyes are not visible
                # baby face down
                if eyesPosition > 0:
                    baby_position = "Face Down"
                # right eyes x value is lower than left eye x value
                # eyesposition will be negative
                # eyes are visible
                # baby face up
                elif eyesPosition < 0:
                    baby_position = "Face Up"
            return baby_position
        else:
            baby_position = "Covered"
            return baby_position


class CameraOps:
    def __init__(self, camSource):
        self.camSource = camSource
        self.cap = None
        self.pose = PoseEstimation()

        self.toStream = True
        self.curFrame = None
        self.warning_message = ""
        self.warning_severity = 'LOW'  ## LOW/MEDIUM/HIGH

        self._yunet = None
        self._yunet_input_size = None  # (w, h)

        self.load_stream()
    def load_stream(self):
        self.cap = cv2.VideoCapture(self.camSource)
        ret, frame = self.cap.read()
        if ret:
            logger.info("Cam/video load successful.")
            return True
        logger.error("Unable to load camera/video stream. Check the source path.")
        return False

# Input is Face status and Body position
    def babyDangerWarning(self, face, body):
        # Face detected cases
        if face == "Face Detected":
            if body == "On It's Side":
                return "Face detected and On It's Side", "MEDIUM"
            if body in ("Covered", "Face Up"):
                return "Face detected and Face Up", "LOW"
            # Face detected but body position is something else / unknown
            return "Face detected", "LOW"

        # Face covered / danger cases
        if face == "DANGER":
            if body == "Face Down":
                return "WARNING baby in DANGER Face Down", "HIGH"
            if body == "Covered":
                return "WARNING baby in DANGER could be Covered", "HIGH"
            if body == "Face Up":
                return "DANGER Face Covered body Face Up Position", "HIGH"
            # Danger but body unknown
            return "WARNING baby in DANGER", "HIGH"

        # Default fallback: no baby / unknown state
        return "No Baby Found", "HIGH"

    def start_cam_stream(self):
        logger.info("Started camera stream")
        self.toStream = True
        babyDanger = 0
        babyOK = 0
        # =================================================================================================
        while self.toStream:
            # print("Streaming...")
            success, image = self.cap.read()
            # True for night video, False for day video
            night = False
            if not success:
                logger.warning("No frames to track.")
                self.toStream=False
                break
            # =================================================================================================
            # Commented out drawing landmarks on the frame to improve performance
            # while testing different models for face detection
            # Also it's just a tool for visual aid of what's happening, won't be used in final program
            # bodyTrack = pose.drawLandmarks(image)
            # pose.drawLandmarks(image)
            # =================================================================================================
            # - BGR for OpenCV/YuNet
            # - RGB for MediaPipe
            if night:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                bgr_for_yunet = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
                rgb_for_pose = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
            else:
                bgr_for_yunet = image
                rgb_for_pose = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            # YuNet on BGR
            face, frame = self.faceTrackingYuNet(bgr_for_yunet)

            # MediaPipe on RGB
            babyPosition = str(self.pose.babyPosition(rgb_for_pose))
            # Warning message to be displayed on the frame
            msg, sev = self.babyDangerWarning(face, babyPosition)

            if face == "DANGER":
                babyDanger += 1
                if babyOK < babyDanger and babyDanger > 3:
                    self.warning_message, self.warning_severity = msg, sev
                    babyDanger = 0

            elif face == "Face Detected":
                babyOK += 1
                if babyOK > babyDanger and babyOK > 3:
                    self.warning_message, self.warning_severity = msg, sev
                    babyOK = 0

            logger.debug(
                "position=%s face=%s msg=%s sev=%s",
                babyPosition, face, self.warning_message, self.warning_severity
            )
            # =================================================================================================
            # Flip the image horizontally for a selfie-view display.
            # image = cv2.flip(image, 1)
            image = cv2.putText(frame, self.warning_message, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2, cv2.LINE_AA)

            self.curFrame = image
            cv2.imshow('Position', image)
            # cv2.imshow('Body Track', bodyTrack)

            # Wait for 'q' key to stop the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

    def stop_cam_stream(self):
        self.toStream = False

    def visualize(self, image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
        # ====================================================================================================
        # Visualize Method from YuNet demo.py
        # https://github.com/opencv/opencv_zoo/blob/master/models/face_detection_yunet/demo.py
        output = image.copy()
        landmark_color = [
            (255,   0,   0), # right eye
            (  0,   0, 255), # left eye
            (  0, 255,   0), # nose tip
            (255,   0, 255), # right mouth corner
            (  0, 255, 255)  # left mouth corner
        ]
        if fps is not None:
            cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

        for det in (results if results is not None else []):
            bbox = det[0:4].astype(np.int32)
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

            conf = det[-1]
            cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)

            landmarks = det[4:14].astype(np.int32).reshape((5,2))
            for idx, landmark in enumerate(landmarks):
                cv2.circle(output, landmark, 2, landmark_color[idx], 2)

        return output

    def faceTrackingYuNet(self, image):
        h, w = image.shape[:2]
        # Setting confidence Threshold of finding a face to 80% (confThreshold)
        # Setting number of faces to 1 instead of 5000 (topK)
        # Instantiating YuNet model
        if self._yunet is None:
            self._yunet = YuNet(
                modelPath='DataSets/face_detection_yunet_2022mar.onnx',
                inputSize=[320, 320],
                confThreshold=0.80,
                nmsThreshold=0.3,
                topK=1,
                backendId=3,
                targetId=0
            )
            logger.info("YuNet initialized")
        # Inference
        # Update input size only if it changed
        if self._yunet_input_size != (w, h):
            self._yunet.setInputSize([w, h])
            self._yunet_input_size = (w, h)

        results = self._yunet.infer(image)

        # Default for this frame: no face detected
        face = "None"

        # infer() can return None or an empty array when nothing is detected
        if results is None or len(results) == 0:
            results = None
        else:
            # If we got a detection, it has a confidence score in the last element
            conf = float(results[0][-1])
            face = "faceDetected" if conf >= 0.80 else "faceCovered"

        # Draw results on the input image
        frame = self.visualize(image, results)

        # Visualize results in a new Window
        #cv2.imshow('Face Tracking ', frame)

        # Checking to see what state the face is and returning a status message
        # If face is covered or cannot be detected set and return message of DANGER
        if face == "faceCovered" or face == "None":
            message = "DANGER"
        # If face is detected set and return message of Face Detected
        elif face == "faceDetected":
            message = "Face Detected"
        # Else if previous checks fail set and return a message of program having difficulty
        else:
            message = "Having trouble detecting the face"
        return message, frame
    # ====================================================================================================



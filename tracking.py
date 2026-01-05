import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from yunet import YuNet

logger = logging.getLogger(__name__)

class PoseEstimation:

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks: bool = False,
        enable_segmentation: bool = False,
        smooth_segmentation: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.4,
    ):
        # Keep the original parameter names as instance variables to make it easier
        # to tune later (and to see what defaults are used).
        self.mode = static_image_mode
        self.comp = model_complexity
        self.smooth = smooth_landmarks
        self.enSeg = enable_segmentation
        self.smoothSeg = smooth_segmentation
        self.minDetect = min_detection_confidence
        self.minTrack = min_tracking_confidence

        # MediaPipe helpers for drawing + pose model.
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Use keyword args (clearer and more stable across MediaPipe versions)
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.comp,
            smooth_landmarks=self.smooth,
            enable_segmentation=self.enSeg,
            smooth_segmentation=self.smoothSeg,
            min_detection_confidence=self.minDetect,
            min_tracking_confidence=self.minTrack,
        )

    def drawLandmarks(self, image, draw: bool = True):
        results = self.pose.process(image)
        if results.pose_landmarks and draw:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return image

    def babyPosition(self, image):
        # Default to "Covered" until pose landmarks are seen.
        baby_position = "Covered"

        # NOTE: image dimensions are hard-coded here because the original project
        # assumed a 640x480 stream. In a later refactor, use `image.shape`.
        image_height = 480
        image_width = 640

        results = self.pose.process(image.copy())
        if not results.pose_landmarks:
            return baby_position

        # MediaPipe landmarks are normalized [0..1] in X/Y, and Z is roughly in
        # "image width" units. Scale Z by image dims for easier thresholds.
        lefteyeX = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EYE].x
        righteyeX = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EYE].x
        eyesPosition = righteyeX - lefteyeX

        leftShoulderZ = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].z * image_height
        rightShoulderZ = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].z * image_width

        # Side logic:
        # If shoulders have opposite Z signs (one closer, one farther), the baby is
        # likely rotated sideways. The "differenceZ" threshold is a tuned value.
        if (leftShoulderZ < 0 < rightShoulderZ) or (leftShoulderZ > 0 > rightShoulderZ):
            differenceZ = -1 * (leftShoulderZ + rightShoulderZ)
            if differenceZ > 30 or differenceZ < -30:
                return "On It's Side"

        # Face up / face down logic:
        # When shoulders are roughly at the same depth sign, use eye ordering.
        if (leftShoulderZ > 0 and rightShoulderZ > 0) or (leftShoulderZ < 0 and rightShoulderZ < 0):
            if eyesPosition > 0:
                return "Face Down"
            if eyesPosition < 0:
                return "Face Up"

        return baby_position

class CameraOps:

    def __init__(self, camSource):
        self.camSource = camSource
        self.cap = None

        # Pose estimator (MediaPipe)
        self.pose = PoseEstimation()

        # Streaming / display state
        self.toStream = True
        self.curFrame = None

        # Warning state displayed on the frame
        self.warning_message = ""
        self.warning_severity = "LOW"  # LOW / MEDIUM / HIGH

        # YuNet face detector state (initialized lazily)
        self._yunet = None
        self._yunet_input_size = None  # (w, h)

        # Keep model path as a Path object so it can be validated early and print
        # an absolute path on errors.
        self._yunet_model_path = Path("DataSets") / "face_detection_yunet_2022mar.onnx"

        self.load_stream()

    def load_stream(self):
        self.cap = cv2.VideoCapture(self.camSource)
        ret, _frame = self.cap.read()
        if ret:
            logger.info("Cam/video load successful.")
            return True

        logger.error("Unable to load camera/video stream. Check the source path.")
        return False

    # Input is Face status and Body position
    def babyDangerWarning(self, face, body):
        if face == "Face Detected":
            if body == "On It's Side":
                return "Face detected and On It's Side", "MEDIUM"
            if body in ("Covered", "Face Up"):
                return "Face detected and Face Up", "LOW"
            return "Face detected", "LOW"

        if face == "DANGER":
            if body == "Face Down":
                return "WARNING baby in DANGER Face Down", "HIGH"
            if body == "Covered":
                return "WARNING baby in DANGER could be Covered", "HIGH"
            if body == "Face Up":
                return "DANGER Face Covered body Face Up Position", "HIGH"
            return "WARNING baby in DANGER", "HIGH"

        return "No Baby Found", "HIGH"

    def start_cam_stream(self):
        logger.info("Started camera stream")

        # Debug visualization (toggle off for better performance)
        show_pose_landmarks = True

        self.toStream = True

        # Debounce counters: these prevent flickering warnings on noisy detections.
        babyDanger = 0
        babyOK = 0

        while self.toStream:
            success, image = self.cap.read()
            night = False  # Set True if using night vision footage preprocessing

            if not success:
                logger.warning("No frames to track.")
                self.toStream = False
                break

            # Keep two versions:
            # - BGR for OpenCV/YuNet (OpenCV default channel order)
            # - RGB for MediaPipe Pose (MediaPipe expects RGB)
            if night:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                bgr_for_yunet = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
                rgb_for_pose = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
            else:
                bgr_for_yunet = image
                rgb_for_pose = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # YuNet face detection + visualization overlay (BGR)
            face, frame = self.faceTrackingYuNet(bgr_for_yunet)

            # MediaPipe pose classification (RGB)
            babyPosition = self.pose.babyPosition(rgb_for_pose)

            # Compute message/severity once for this frame.
            msg, sev = self.babyDangerWarning(face, babyPosition)

            # Debounce logic: require >3 consecutive frames before updating.
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
                babyPosition,
                face,
                self.warning_message,
                self.warning_severity,
            )

            # Start with the YuNet visualization frame (BGR)
            display_frame = frame

            # Optional debug overlay: blend pose landmarks over the YuNet frame.
            if show_pose_landmarks:
                pose_vis_rgb = self.pose.drawLandmarks(rgb_for_pose.copy(), draw=True)
                pose_vis_bgr = cv2.cvtColor(pose_vis_rgb, cv2.COLOR_RGB2BGR)
                display_frame = cv2.addWeighted(display_frame, 0.7, pose_vis_bgr, 0.3, 0)

            # Draw the warning text
            display_frame = cv2.putText(
                display_frame,
                self.warning_message,
                (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            self.curFrame = display_frame
            cv2.imshow("Position", display_frame)

            # Quit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop_cam_stream(self):
        self.toStream = False

    def visualize(self, image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
        output = image.copy()
        landmark_color = [
            (255, 0, 0),    # right eye
            (0, 0, 255),    # left eye
            (0, 255, 0),    # nose tip
            (255, 0, 255),  # right mouth corner
            (0, 255, 255),  # left mouth corner
        ]

        if fps is not None:
            cv2.putText(
                output,
                "FPS: {:.2f}".format(fps),
                (0, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
            )

        for det in (results if results is not None else []):
            bbox = det[0:4].astype(np.int32)
            cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)

            conf = det[-1]
            cv2.putText(
                output,
                "{:.4f}".format(conf),
                (bbox[0], bbox[1] + 12),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                text_color,
            )

            landmarks = det[4:14].astype(np.int32).reshape((5, 2))
            for idx, landmark in enumerate(landmarks):
                cv2.circle(output, landmark, 2, landmark_color[idx], 2)

        return output

    def faceTrackingYuNet(self, image):
        h, w = image.shape[:2]

        # create the detector once (big performance win vs creating per frame).
        if self._yunet is None:
            if not self._yunet_model_path.exists():
                raise FileNotFoundError(f"YuNet model not found: {self._yunet_model_path.resolve()}")

            self._yunet = YuNet(
                modelPath=str(self._yunet_model_path),
                inputSize=[320, 320],
                confThreshold=0.80,
                nmsThreshold=0.3,
                topK=1,
                backendId=3,
                targetId=0,
            )
            logger.info("YuNet initialized")

        # YuNet requires the current input size. Update only when the size changes.
        if self._yunet_input_size != (w, h):
            self._yunet.setInputSize([w, h])
            self._yunet_input_size = (w, h)

        results = self._yunet.infer(image)

        # Default: no face detected in this frame.
        face = "None"

        # infer() can return None or an empty array when nothing is detected.
        if results is None or len(results) == 0:
            results = None
        else:
            # Confidence is the last element in YuNet output. Use an explicit threshold
            # (don't rely on float truthiness).
            conf = float(results[0][-1])
            face = "faceDetected" if conf >= 0.80 else "faceCovered"

        frame = self.visualize(image, results)

        # Map internal state to the message string the rest of the pipeline expects.
        if face in ("faceCovered", "None"):
            message = "DANGER"
        elif face == "faceDetected":
            message = "Face Detected"
        else:
            message = "Having trouble detecting the face"

        return message, frame

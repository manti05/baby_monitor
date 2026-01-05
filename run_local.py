import argparse
from tracking import CameraOps
import logging

logging.basicConfig(level=logging.INFO)

def parse_source(value: str):
    # Allow: "0" -> webcam, or "demoNight.avi" -> file
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="Run CV tracking locally with OpenCV window output.")
    parser.add_argument("--source", default="0", help='Video source: "0" for webcam or path/URL to video stream')
    args = parser.parse_args()

    cam = CameraOps(camSource=parse_source(args.source))
    cam.start_cam_stream()


if __name__ == "__main__":
    main()

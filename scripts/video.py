import pathlib
import os

import cv2

GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_DIR = GIT_ROOT / "data"

def main():
    cap = cv2.VideoCapture(str(DATA_DIR / 'embodied_learning' / 'block-a-blue-day1-first-group-cam2.mp4'))

    while True:
        ret, frame = cap.read()
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
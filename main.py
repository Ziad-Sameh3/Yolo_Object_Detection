import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
import cv2

import cv2


# load yolov8 model
model = YOLO("best.pt")

# load video
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)


# Set up video writer to save output video
output_path = "output_video.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


frame_count = 1
frame_skip_rate = 1
ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        if frame_count % frame_skip_rate == 0:

            # track objects
            results = model.track(frame, persist=True)

            frame_ = results[0].plot()

            out.write(frame_)

            # visualize
        cv2.imshow("frame", frame_)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    # frame_count += 1

# Release resources
cap.release()
out.release()  # Save the output video
cv2.destroyAllWindows()

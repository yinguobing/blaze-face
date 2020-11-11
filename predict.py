"""Sample module for predicting face marks with HRNetV2."""
from argparse import ArgumentParser

import cv2
import numpy as np
import tensorflow as tf

from anchors import Anchors
from postprocessing import decode, draw_face_boxes
from preprocessing import normalize

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


if __name__ == "__main__":
    # Restore the model.
    model = tf.keras.models.load_model("./exported")

    model.summary()

    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        frame = frame[:960, :960]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Read in and preprocess the sample image
        image = cv2.resize(frame, (128, 128))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_input = normalize(np.array(img_rgb, dtype=np.float32))

        # Do prediction.
        prediction = model.predict(tf.expand_dims(img_input, 0))[0]

        # Parse the prediction to get face locations.
        faces = decode(prediction, 0.2)

        # Draw the face boxes.
        draw_face_boxes(frame, faces)

        # Show the result in windows.
        cv2.imshow('image', frame)
        if cv2.waitKey(1) == 27:
            break

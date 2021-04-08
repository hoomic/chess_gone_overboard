import chess
from chesscv import ChessBoard

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 as cv

camera = PiCamera(resolution=(720,720))
rawCapture = PiRGBArray(camera)
board_tracker = ChessBoard(display=True)
time.sleep(2.0)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
  image = cv.resize(frame.array, (300, 300), interpolation=cv.INTER_AREA)
  board_tracker.locate_squares(image)
  board_tracker.overlay_grid(image)

  rawCapture.truncate(0)



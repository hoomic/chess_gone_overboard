import chess
from chesscv import ChessBoard

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2 as cv

camera = PiCamera()
camera.resolution(300, 300)
rawCapture = PiRGBArray(camera)
time.sleep(0.1)
board_tracker = ChessBoard()
while True:
  camera = capture(rawCapture, format='bgr')
  image = rawCapture.array
  board_tracker.locate_squares(image)
  board_tracker.overlay_grid(image)



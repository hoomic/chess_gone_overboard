import chess
from chesscv import BoardDetector, show_wait_destroy

import os
# if this is running on a pi, then import picamera
if os.uname()[4][:3].startswith('arm'):
  from picamera.array import PiRGBArray
  from picamera import PiCamera
import time
import cv2 as cv
import numpy as np

class InvalidMove(Exception):
  pass

def track_game(video):
  board_detector = BoardDetector(display=True)
  move_tracker = MoveTracker()

  if video is None:
    camera = PiCamera(resolution=(720,720))
    rawCapture = PiRGBArray(camera)
    time.sleep(2.0)
    camera.capture(rawCapture, format="bgr")
    first_frame = rawCapture.array
    params = board_detector.locate_squares(first_frame)
    move_tracker.set_square_coordinates(params)
    rawCapture.truncate(0)
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
      frame = board_detector.transform_image(frame)
      move_tracker.process_image(frame)

      rawCapture.truncate(0)
  else:
    vs = cv.VideoCapture(video)
    board_found = False
    while not board_found:
      _, first_frame = vs.read()
      try:
        params = board_detector.locate_squares(first_frame)
        move_tracker.set_square_coordinates(params)
        board_found = True
      except TypeError:
        continue
    while (vs.isOpened()):
      ret, frame = vs.read()
      if not ret:
        break
      frame = board_detector.transform_image(frame)
      move_tracker.process_image(frame)

class Square():
  """
  object for a single square on the board
  """
  def __init__(self, board_coords, image_coords):
    self.board_coords = board_coords
    self.image_coords = image_coords
    self.piece = '.'

  def detect_difference(self, prev_image, curr_image):
    last_square = self.get_image(prev_image)
    square = self.get_image(curr_image)
    delta = cv.absdiff(last_square, square)
    if np.mean(delta) > 25:
      return self.board_coords
    return None

  def get_image(self, image):
    x0, x1, y0, y1 = self.image_coords
    return image[y0:y1,x0:x1]

  def display(self, image):
    square = self.get_image(image)
    show_wait_destroy("{},{}".format(*self.board_coords), square)

class MoveTracker():
  """
  Class to detect moves made on the board
  """
  def __init__(self):
    self.last_still_frame = None
    self.prev_frame = None
    self.board = chess.Board()
    self.white_side = None

  def process_image(self, frame):
    #frame = cv.GaussianBlur(frame, (5, 5), 0)

    if self.prev_frame is None:
      self.prev_frame = frame
      return

    # compute the absolute difference between the previous frame and the current frame
    frame_delta = cv.absdiff(self.prev_frame, frame)
    if np.all(frame_delta < 40): #image is still
      #show_wait_destroy('still', frame)
      if self.last_still_frame is None:
        self.last_still_frame = frame
        return
      # so see if the board differs from the last still board
      square_changes = []
      for square in self.squares:
        diff = square.detect_difference(self.last_still_frame, frame)
        if diff is not None:
          square_changes.append(diff)
      if not len(square_changes):
        return
      elif len(square_changes) == 1:
        # When a move occurs, at least two squares should change
        import pdb; pdb.set_trace()
        return
      elif len(square_changes) > 4:
        # If more than 4 squares change, this is also not possible
        #import pdb; pdb.set_trace()
        return
      else:
        move = self.process_move(square_changes)
        #if move is not None:
        #  show_wait_destroy(move.uci(), frame)
        print(self.board)
        print('-' * 50)
      self.last_still_frame = frame
    self.prev_frame = frame

  def process_move(self, square_changes):
    if self.white_side is None:
      #this should be the first move
      if np.all([s[0] <= 4 for s in square_changes]):
        self.white_side = 'left'
      elif np.all([s[0 > 4] for s in square_changes]):
        self.white_side = 'right'
      else:
        import pdb; pdb.set_trace()
    move = None
    if len(square_changes) == 2: # "normal" move
      s0 = self.get_square(square_changes[0])
      s1 = self.get_square(square_changes[1])
      try:
        move = self.board.find_move(s0, s1)
      except ValueError:
        move = self.board.find_move(s1, s0)
    elif len(square_changes) == 3: # en passant move
      if not self.board.has_legal_en_passant():
        raise InvalidMove("En passant attempted when not legal")
      pass
    elif len(square_changes) == 4: # castle
      pass
    if move is None:
      return
    self.board.push(move)
    return move

  def get_square(self, square_id):
    if self.white_side == 'left':
      return chess.square(square_id[1], square_id[0])
    else:
      return chess.square(7-square_id[1], 7-square_id[0])

  def set_square_coordinates(self, params):
    self.squares = []
    x, y ,s = params
    for i in range(8):
      x_lo = i * s
      x_hi = (i + 1) * s
      for j in range(8):
        y_lo = j * s
        y_hi = (j + 1) * s
        self.squares.append(Square((i, j), (x_lo, x_hi, y_lo, y_hi)))


if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description='Track a chess game over a live board')
  parser.add_argument('--video', dest='video', default=None, help='path to video file')

  args = parser.parse_args()

  track_game(args.video)



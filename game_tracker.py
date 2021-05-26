import chess
from chess import pgn
from chesscv import BoardDetector, show_wait_destroy
from piece_classifier import train_piece_classifier, get_piece_prediction

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

def track_game(video, outfile):
  board_detector = BoardDetector(display=True)
  move_tracker = MoveTracker()

  if video is None:
    camera = PiCamera(resolution=(720,720))
    rawCapture = PiRGBArray(camera)
    time.sleep(2.0)
    camera.capture(rawCapture, format="bgr")
    first_frame = rawCapture.array
    params, ps = board_detector.locate_squares(first_frame)
    move_tracker.set_square_coordinates(params, ps)
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
        params, ps = board_detector.locate_squares(first_frame)
        move_tracker.set_square_coordinates(params, ps)
        move_tracker.set_first_frame(board_detector.transform_image(first_frame))
        board_found = True
      except TypeError:
        continue
    classifier_trained = False
    while (vs.isOpened()):
      ret, frame = vs.read()
      if not ret:
        break
      cv.imshow('frame', frame)
      frame = board_detector.transform_image(frame)
      if cv.waitKey(100) & 0xFF == ord('q'):
        break
      if not classifier_trained:
        move_tracker.train_piece_classifier(frame)
        classifier_trained = True
      move_tracker.process_image(frame)
    output_game_to_file(move_tracker.board, outfile)

def output_game_to_file(board, outfile):
  game = pgn.Game()
  node = None
  for move in board.move_stack:
    if node is None:
      node = game.add_variation(move)
    else:
      node = node.add_variation(move)
  print(game, file=open(outfile, 'w'), end='\n\n')

class Square():
  """
  object for a single square on the board
  """
  def __init__(self, board_coords, image_coords):
    self.board_coords = board_coords
    self.image_coords = image_coords

  def get_delta(self, prev_frame, curr_frame):
    prev_image = self.get_image(prev_frame)
    curr_image = self.get_image(curr_frame)
    return cv.absdiff(prev_image, curr_image)

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
    first_delta = cv.absdiff(self.last_still_frame, frame)
    if np.all(frame_delta < 40) and np.percentile(first_delta, 90) < 50: #image is still
      #show_wait_destroy('still: {0:.2f}'.format(np.percentile(first_delta, 90)), frame)
      # so see if the board differs from the last still board
      move_candidates = []
      for move in self.board.legal_moves:
        if self.white_side is None:
          #first calculate as if it is left side, then right side
          from_square = self.get_square(move.from_square, 'left') 
          to_square = self.get_square(move.to_square, 'left')
          if self.detect_move(from_square, to_square, frame):
            self.white_side = 'left'
            print("White side is left")
            move_candidates.append(move)
          from_square = self.get_square(move.from_square, 'right')
          to_square = self.get_square(move.to_square, 'right')
          if self.detect_move(from_square, to_square, frame):
            self.white_side = 'right'
            print("White is is right")
            move_candidates.append(move)
        else:
          from_square = self.get_square(move.from_square)
          to_square = self.get_square(move.to_square)
          if self.detect_move(from_square, to_square, frame):
            move_candidates.append(move)
      move_made = None
      if len(move_candidates) == 1:
        move_made = move_candidates[0]
      elif len(move_candidates) > 1:
        move_made = self.check_for_castle(move_candidates)
        if not move_made:
          move_made = self.filter_multiple_moves(move_candidates, frame)
          if not move_made: 
            import pdb; pdb.set_trace()
      if move_made is not None:
        self.board.push(move_made)
        print(self.board)
        print('-' * 50)
        show_wait_destroy(move_made.uci(), frame)
        self.last_still_frame = frame
    self.prev_frame = frame

  def check_for_castle(self, move_candidates):
    if self.board.turn == chess.WHITE:
      if chess.Move.from_uci('e1g1') in move_candidates:
        return chess.Move.from_uci('e1g1')
      elif chess.Move.from_uci('e1c1') in move_candidates:
        return chess.Move.from_uci('e1c1')
    else:
      if chess.Move.from_uci('e8g8') in move_candidates:
        return chess.Move.from_uci('e8g8')
      elif chess.Move.from_uci('e8c8') in move_candidates:
        return chess.Move.from_uci('e8c8')
    return None

  def detect_move(self, from_square, to_square, frame):
    from_pred = self.get_piece_prediction(from_square.get_image(frame))
    to_pred = self.get_piece_prediction(to_square.get_image(frame))
    from_delta = from_square.get_delta(self.last_still_frame, frame)
    to_delta = to_square.get_delta(self.last_still_frame, frame)
    if np.median(from_delta) < 10 or np.median(to_delta) < 10:
      return False
    if from_pred < -2 and to_pred > 0:
#      cv.imshow('from_curr: {0:.3f}'.format(from_pred), from_square.get_image(frame))
#      show_wait_destroy('to_curr: {0:.3f}'.format(to_pred), to_square.get_image(frame))
      return True
    # if the from square definitely doesn't have a piece, then take a closer look at
    # the to squares
    elif from_pred < -3 and np.median(from_delta) > 25: 
      print("Taking a closer look at to_square")
      to_pred = self.get_piece_prediction(to_square.get_image(frame), True)
#      cv.imshow('from_curr: {0:.3f}'.format(from_pred), from_square.get_image(frame))
#      show_wait_destroy('to_curr: {0:.3f}'.format(to_pred), to_square.get_image(frame))
      if to_pred > 0 and np.median(to_delta) > 25:
        return True
    return False

  def filter_multiple_moves(self, move_candidates, frame):
    # if all moves have the same from_square, then just look at the to squares
    if np.all([m.from_square == move_candidates[0].from_square for m in move_candidates]):
      deltas = []
      for move in move_candidates:
        to_square = self.get_square(move.to_square)
        to_delta = to_square.get_delta(self.last_still_frame, frame)
        deltas.append(np.median(to_delta))
      # if one delta median is 2x greater than the rest, then chose that one
      if np.all([np.max(deltas) > 2*d for i, d in enumerate(deltas) if i != np.argmax(deltas)]):
        return move_candidates[np.argmax(deltas)]

  def get_square(self, square_id, side=None):
    if side is None:
      side = self.white_side
    if side == 'left':
      return self.squares[square_id]
    else:
      return self.squares[63 - square_id]

  def set_square_coordinates(self, params, perspective_shift):
    self.squares = []
    x, y ,s = params
    p0 = perspective_shift[0]
    for i in range(8):
      p_i = min(i, 7-i)
      lo_shift = -perspective_shift[p_i] if i < 4 else perspective_shift[p_i + 1]
      hi_shift = -perspective_shift[p_i + 1] if i < 4 else perspective_shift[p_i]
      x_lo = i * s + lo_shift + p0
      x_hi = (i + 1) * s + hi_shift + p0
      for j in range(8):
        p_j = min(j, 7-j)
        lo_shift = -perspective_shift[p_j] if j < 4 else perspective_shift[p_j + 1]
        hi_shift = -perspective_shift[p_j + 1] if j < 4 else perspective_shift[p_j]
        y_lo = j * s + lo_shift + p0
        y_hi = (j + 1) * s + hi_shift + p0
        self.squares.append(Square((i, j), (x_lo, x_hi, y_lo, y_hi)))

  def set_first_frame(self, frame):
    self.last_still_frame = frame
  
  def train_piece_classifier(self, frame):
    images = []
    labels = []
    for i, square in enumerate(self.squares):
      try:
        image = cv.resize(square.get_image(frame), (32, 32), interpolation=cv.INTER_AREA)
      except:
        import pdb; pdb.set_trace()
      image = image/255
      label = 1 if i < 16 or i > 47 else 0
      for image in generate_transformations(image):
        images.append(image)
        labels.append(label)
    images = np.array(images)
    images = images.reshape(-1, 1, 32, 32)
    labels = np.array(labels)
    labels = labels.reshape(-1, 1)
    self.piece_classifier = train_piece_classifier(images, labels)

  def get_piece_prediction(self, orig_image, transform = False):
    image = np.copy(orig_image)
    image = cv.resize(image, (32, 32), interpolation=cv.INTER_AREA)
    image = image/255
    if transform:
      preds = []
      for image in generate_transformations(image, True):
        image = image.reshape(1, 1, 32, 32)
        pred = get_piece_prediction(self.piece_classifier, image)
        preds.append(get_piece_prediction(self.piece_classifier, image))
      return np.max(preds)
    else:
      image = image.reshape(1, 1, 32, 32)
      return get_piece_prediction(self.piece_classifier, image)

def generate_transformations(image, roll=False):
  """ function to make mirror and rotation transformations """
  for j in range(4):
    new_image = np.copy(image)
    rotation_matrix = cv.getRotationMatrix2D((16, 16), j * 90, 1.0)
    new_image = cv.warpAffine(new_image, rotation_matrix, (32, 32))
    yield new_image
  new_image = np.copy(image)
  new_image = cv.flip(new_image, 0)
  for j in range(4):
    new_image = np.copy(new_image)
    rotation_matrix = cv.getRotationMatrix2D((16, 16), j * 90, 1.0)
    new_image = cv.warpAffine(new_image, rotation_matrix, (32, 32))
    yield new_image
  if roll:
    for i in range(6):
      new_image = np.copy(image)
      new_image = np.roll(new_image, i, 0)
      for j in range(6):
        new_image = np.roll(new_image, j, 1)
        yield new_image
  

if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description='Track a chess game over a live board')
  parser.add_argument('-v', '--video', dest='video', default=None, help='path to video file')
  parser.add_argument('-o', '--outfile', dest='outfile', default="./game.pgn"
    , help='path to output pgn file')

  args = parser.parse_args()

  track_game(args.video, args.outfile)



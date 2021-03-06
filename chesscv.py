"""
@file chesscv.py
@brief 
"""
import numpy as np
import sys
import cv2 as cv
import chess
from collections import defaultdict

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def rotate_image(image, angle):
    (cX, cY) = (image.shape[0] // 2, image.shape[1] // 2)
    rotation_matrix = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotate = np.copy(image)
    return cv.warpAffine(rotate, rotation_matrix, (image.shape[0], image.shape[1]))

def generate_grid_coordinates(x, y, s):
  coordinates = []
  for i in range(9):
    x0 = x + i * s
    for j in range(9):
      coordinates.append((x0, y + j * s))
  return coordinates

def generate_grids(center_corners, square_length):
  for Cx, Cy in center_corners:
    for s in range(square_length - 2, square_length + 3):
      for i in range(2,5):
        x = Cx - i * s
        for j in range(2,5):
          y = Cy - j * s
          yield generate_grid_coordinates(x, y, s), (x, y, s)

def get_center(image):
  center = np.copy(image)
  dim = center.shape[0]
  third = int(dim / 3)
  return center[third:dim - third, third:dim - third]

def overlap_match_score(corners, grid, params):
  corners = np.array(corners)
  grid = np.array(grid)
  distance_matrix = np.linalg.norm(corners[:, None, :] - grid[None, :, :], axis=-1)
  min_distances = np.min(distance_matrix, axis=0)
  return np.mean(min_distances)

class BoardDetector():
  def __init__(self, resize=300, display=False):
    self.resize = resize
    self.display = display

  def locate_squares(self, src):
    src = cv.resize(src, (self.resize, self.resize), interpolation=cv.INTER_AREA)

    if self.display:
      # Show source image
      cv.imshow("src", src)

    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
      gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
      gray = src
    # if the image is too dark, wait for camera to adjust

    if np.mean(gray) < 30:
      return None

    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)

    #increase contrast using CLAHE
    tile_size = self.resize // 100
    self.clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(tile_size, tile_size))
    gray = self.clahe.apply(gray)
    gray = cv.bilateralFilter(gray,5,50,50)
    if self.display:
      show_wait_destroy("clahe", gray)

    # get the center 9th of the board
    center = get_center(gray)

    # get the board angle
    self.set_camera_angle(-np.degrees(self.get_board_angle(center, 50)))
    
    #rotate the image by that angle
    gray = self.rotate_image(gray)
    center = rotate_image(center, self.camera_angle)

    #find the best guess at square lengths 
    square_length, center_corners = self.get_square_length(center)
    self.side_length = int(square_length * 8)

    #find all corners on the board
    all_corners = self.detect_corners(gray, square_length)
    
    grids = []
    best_grid = None
    best_params = None
    best_score = 1e6
    for grid, params in generate_grids(center_corners, square_length):
      if grid is None:
        continue 
      score = overlap_match_score(all_corners, grid, params)
      if score < best_score:
        best_grid = grid
        best_params = params
        best_score = score
      grids.append((score, grid, params))
    ps = self.get_perspective_shift(gray, best_params)
    self.best_grid = best_grid
    if self.display:
      self.overlay_grid(gray, best_params, ps)
    x, y, s = best_params
    self.board_coords = (x - ps[0], x + 8 * s + ps[0], y - ps[0], y + 8 * s + ps[0])
    return best_params, ps

  def get_board_angle(self, orig_gray, threshold):
    """ 
    Finds the angular orientation of the board by looking at the center 9th
    of the image and detecting all the lines in that image. Filter for lines
    that are approximately vertical or horizontal and then convert the vertical
    angles into horizontal ones. Collect all angles and use the median angle as
    the best guess for the angle of the board
    """
    gray = np.copy(orig_gray)
    edges = np.copy(gray)
    #first detect edges using Canny
    edges = cv.Canny(edges, 100, 150, apertureSize = 3)
    if self.display:
      cv.imshow("edges", edges)
    # then detect lines using a Hough transform
    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=threshold, minLineLength=10, maxLineGap=10)
    thetas = []
    for line in lines:
      x1, y1, x2, y2 = line[0]
      # calculate the angle of every line
      theta = np.arctan(np.divide(y2 - y1, x2 - x1))
      # There may be some noise, so filter all lines that aren't approximately vertical or horizontal
      if abs(theta) > 0.2 and abs(theta + np.pi/2) > 0.2:
        continue
      if self.display:
        cv.line(gray, (x1, y1), (x2, y2), 255, 1)
      # Convert all vertical angles into horizontal angles
      if abs(theta) > 0.2: # theta ~= -pi/2
        theta += np.pi/2
      thetas.append(theta)
    # if we didn't find any lines, then try again with half the threshold
    if not len(thetas):
      return get_hough_lines(gray, threshold//2)
    if self.display:
      show_wait_destroy("center_lines", gray)
    # The Hough transforma measures the angle in the opposite direction of 
    # generate_grid, so take the negation here
    return -np.median(thetas)

  def get_square_length(self, orig_gray):
    """
    Finds the length/width of each square by looking at the center 9th of the
    image and detecting the 15 pixels that are most likely to be corners of 
    a square. We assume that the board takes up more than 40% of each dimension
    and that it fits within the image. This function guesses that the side length
    of a square is the median of all distances that are found within greater than
    1/20 of the image (8/20=40%) and less than 1/8 of the board (8/8=100%)
    """
    gray = np.copy(orig_gray)
    # detect 15 most likely corners
    corners = cv.goodFeaturesToTrack(gray, 20, 0.0001, self.resize // 20)
    corners = np.int0(corners)
    # calculate distance from each corner to every other corner
    distances = []
    filtered_corners = []
    lo = self.resize // 20
    hi = (self.resize // 8) + 1
    for i, c0 in enumerate(corners):
      x0, y0 = c0.ravel()
      # filter out corners at edge of board
      if x0 < 5 or x0 > (self.resize // 3) - 5:
        continue
      if y0 < 5 or y0 > (self.resize // 3) - 5:
        continue
      if self.display:
        cv.circle(gray, (x0, y0), 3, 255, -1)
      for j, c1 in enumerate(corners):
        if j < i:
          continue
        x1, y1 = c1.ravel()
        # filter out corners at edge of board
        if x1 < 5 or x1 > (self.resize // 3) - 5:
          continue
        if y1 < 5 or y1 > (self.resize // 3) - 5:
          continue
        # filter for distances between lo and hi pixels
        x_dist = np.abs(x0 - x1)
        if x_dist > lo and x_dist < hi:
          distances.append(x_dist)
          filtered_corners.append((x0 + self.resize // 3, y0 + self.resize // 3))
        y_dist = np.abs(y0 - y1)
        if y_dist > lo and y_dist < hi:
          distances.append(y_dist)
          filtered_corners.append((x0 + self.resize // 3, y0 + self.resize // 3))
    if self.display:
      show_wait_destroy("center corners", gray)
      gray = np.copy(orig_gray)
      for x, y in filtered_corners:
        cv.circle(gray, (x - self.resize // 3, y - self.resize // 3), 3, 255, -1)
      show_wait_destroy("filtered_corners", gray)
    # take median of distances and multiply by 8 for entire board side length
    return int(np.median(distances)), list(set(filtered_corners))

  def detect_corners(self, gray, square_length):
    """
    Detect all corners in the grayscale image, and make sure they are all at least
    square_length - 1 apart.
    """
    image = np.copy(gray)
    corners = cv.goodFeaturesToTrack(gray, 120, 0.001, square_length - 1)
    corners = np.int0(corners)
    all_corners = []
    tan_theta = np.tan(np.radians(self.camera_angle))
    border = 3
    for i, c in enumerate(corners):
      x, y = c.ravel()
      if x <= max(border, (y-self.resize/2) * tan_theta + border):
        continue
      elif x >= min(self.resize - border, self.resize - (self.resize/2 - y) * tan_theta - border):
        continue
      elif y <= max(border, (x - self.resize/2) * tan_theta + border):
        continue
      elif y >= min(self.resize - border, self.resize - (self.resize/2 - x) * tan_theta - border):
        continue
      all_corners.append((x, y))
      if self.display:
        cv.circle(image, (x, y), 2, 255, -1)
    if self.display:
      show_wait_destroy("all_corners", image)
    return all_corners

  def get_perspective_shift(self, gray, board_params):
    """ 
    Since the center of the board is closer to the camera than the rest of the board,
    outer squares can be more dilated than the center squares. This function uses the empty
    squares on ranks 3:6 to determine how many more pixels to include for outer squares 
    based on how far they are from the center.
    """
    x, y, s = board_params
    #first look at squares that are 2 in from the edge. c3:c6 and f3:f6
    perspective_shift = defaultdict(int)
    for j in reversed(range(4)):
      shift_values = []
      for i in range(2, 6):
        x_lo = x + i * s
        x_hi = x + (i+1) * s
        for upper in [True, False]:
          y_lo = y + j * s + perspective_shift[j+1] if upper else y + (7-j) * s
          y_hi = y + (j+1) * s - perspective_shift[j+1] if upper else y + (8-j) * s
          line_averages = []
          for y0 in range(y_lo, y_hi):
            line_averages.append(np.mean(gray[y0, x_lo:x_hi]))
          mean = np.mean(line_averages)
          std = np.std(line_averages)
          for k in range(10):
            if abs((np.mean(gray[(y_lo -k) if upper else (y_hi + k), x_lo:x_hi]) - mean) / std) > 2.0:
              shift_values.append(k)
              break
      perspective_shift[j] = int(np.mean(shift_values))
    return perspective_shift

  def set_camera_angle(self, angle):
    (cX, cY) = (self.resize // 2, self.resize // 2)
    self.rotation_matrix = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    self.camera_angle = angle

  def rotate_image(self, image):
    rotate = np.copy(image)
    return cv.warpAffine(rotate, self.rotation_matrix, (self.resize, self.resize))

  def crop_to_board(self, image):
    x_lo, x_hi, y_lo, y_hi = self.board_coords
    return image[y_lo:y_hi, x_lo:x_hi]

  def overlay_grid(self, image, params, perspective_shift):
    overlay = np.copy(image)
    x, y ,s = params
    for i in range(8):
      p_i = min(i, 7-i)
      lo_shift = -perspective_shift[p_i] if i < 4 else perspective_shift[p_i + 1]
      hi_shift = -perspective_shift[p_i + 1] if i < 4 else perspective_shift[p_i]
      x_lo = x + i * s + lo_shift
      x_hi = x + (i + 1) * s + hi_shift
      for j in range(8):
        p_j = min(j, 7-j)
        lo_shift = -perspective_shift[p_j] if j < 4 else perspective_shift[p_j + 1]
        hi_shift = -perspective_shift[p_j + 1] if j < 4 else perspective_shift[p_j]
        y_lo = y + j * s + lo_shift
        y_hi = y + (j + 1) * s + hi_shift
        cv.circle(overlay, (x_lo, y_lo), 3, 255, -1)
        cv.circle(overlay, (x_lo, y_hi), 3, 255, -1)
        cv.circle(overlay, (x_hi, y_lo), 3, 255, -1)
        cv.circle(overlay, (x_hi, y_hi), 3, 255, -1)
    show_wait_destroy('overlay', overlay)

  def transform_image(self, image, gray=True):
    if len(image.shape) != 2 and gray:
      image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (self.resize, self.resize), interpolation=cv.INTER_AREA)
    if gray:
      image = self.clahe.apply(image)
    image = self.rotate_image(image)
    image = self.crop_to_board(image)
    return image

if __name__ == "__main__":
  # [load_image]
  # Check number of arguments
  argv = sys.argv[1:]
  if len(argv) < 1:
    print ('Not enough parameters')
    print ('Usage:\nchesscv.py < path_to_image >')
  # Load the image
  src = cv.imread(argv[0], cv.IMREAD_COLOR)
  # Check if image is loaded fine
  if src is None:
    print ('Error opening image: ' + argv[0])
  display = False
  if len(argv) == 2:
    display = argv[1] == "--display"
  chessboard = BoardDetector(300, display)
  chessboard.locate_squares(src)

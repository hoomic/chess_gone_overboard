"""
@file chesscv.py
@brief 
"""
import numpy as np
import sys
import cv2 as cv

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

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
      for i in range(4):
        x = Cx - i * s
        for j in range(4):
          y = Cy - j * s
          yield generate_grid_coordinates(x, y, s), (x, y, s)

def get_center(image):
  center = np.copy(image)
  dim = center.shape[0]
  third = int(dim / 3)
  return center[third:dim - third, third:dim - third]

def variance_match_score(image, grid, params):
  x, y, s = params
  dark_pixels = []
  light_pixels = []
  for i in range(9):
    x0 = (x + i * (s // 8)) + 5
    x1 = (x + (i + 1) * (s // 8)) - 5
    for j in range(9):
      y0 = (y + j * (s // 8)) + 5
      y1 = (y + (j + 1) * (s // 8)) - 5
      if i % 2 == j % 2:
        light_pixels.extend(list(image[y0:y1,x0:x1].flatten()))
      else:
        dark_pixels.extend(list(image[y0:y1,x0:x1].flatten()))
  return np.var(light_pixels) + np.var(dark_pixels)

def overlap_match_score(corners, grid, params):
  corners = np.array(corners)
  grid = np.array(grid)
  distance_matrix = np.linalg.norm(corners[:, None, :] - grid[None, :, :], axis=-1)
  min_distances = np.min(distance_matrix, axis=0)
  return np.mean(min_distances)

class ChessBoard():
  def __init__(self, resize=300, display=False):
    self.resize = resize
    self.display = display

  def get_space_coordinates(self, src):
    src = cv.resize(src, (self.resize, self.resize), interpolation=cv.INTER_AREA)
    overlay = np.copy(src)

    if self.display:
      # Show source image
      cv.imshow("src", src)

    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
      gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
      gray = src
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = cv.bitwise_not(gray)

    #increase contrast using CLAHE
    tile_size = self.resize // 100
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(tile_size, tile_size))
    gray = clahe.apply(gray)
    gray = cv.bilateralFilter(gray,5,50,50)
    if self.display:
      show_wait_destroy("clahe", gray)

    # get the center 9th of the board
    center = get_center(gray)

    # get the board angle
    self.set_camera_angle(-np.degrees(self.get_board_angle(center, 50)))
    
    #rotate the image by that angle
    gray = self.rotate_image(gray)
    center = self.rotate_image(center)

    #find the best guess at square lengths 
    square_length, center_corners = self.get_board_side_length(center)
    self.side_length = int(square_length * 8)

    #find all corners on the board
    all_corners = self.detect_corners(gray, square_length)

    print("THETA", self.camera_angle)
    print("SIDE LENGTH", self.side_length)
    
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
    overlay = np.copy(gray)
    for x, y in best_grid:
      cv.circle(overlay, (x, y), 3, 255, -1)
    show_wait_destroy('best', overlay)
    return

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

  def get_board_side_length(self, orig_gray):
    """
    Finds the side length of the board by looking at the center 9th of the
    image and detecting the 15 pixels that are most likely to be corners of 
    a square. We assume that the board takes up more than 40% of each dimension
    and that it fits within the image. This function guesses that the side length
    of a square is the median of all distances that are found within 15 and 38 pixels
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
    return int(np.median(distances)), filtered_corners

  def detect_corners(self, gray, square_length):
    """
    Detect all corners in the grayscale image, and make sure they are all at least
    square_length - 1 apart.
    """
    image = np.copy(gray)
    corners = cv.goodFeaturesToTrack(gray, 120, 0.001, square_length - 1)
    corners = np.int0(corners)
    all_corners = []
    for i, c in enumerate(corners):
      x, y = c.ravel()
      all_corners.append((x, y))
      if self.display:
        cv.circle(image, (x, y), 2, 255, -1)
    if self.display:
      show_wait_destroy("all_corners", image)
    return all_corners

  def set_camera_angle(self, angle):
    (cX, cY) = (self.resize // 2, self.resize // 2)
    self.rotation_matrix = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
    self.camera_angle = angle

  def rotate_image(self, image):
    rotate = np.copy(image)
    return cv.warpAffine(rotate, self.rotation_matrix, (self.resize, self.resize))

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
  chessboard = ChessBoard(300, display)
  chessboard.get_space_coordinates(src)

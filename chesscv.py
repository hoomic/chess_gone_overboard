"""
@file chesscv.py
@brief Use morphology transformations for extracting horizontal and vertical lines to
       detect the spaces on a chess board
"""
import numpy as np
import sys
import cv2 as cv

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def get_space_coordinates(src):
  src = cv.resize(src, (300, 300), interpolation=cv.INTER_AREA)
  overlay = np.copy(src)

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
  clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(30,30))
  gray = clahe.apply(gray)
  show_wait_destroy("clahe", gray)

  bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                            cv.THRESH_BINARY, 5, -2)

  # Create the images that will use to extract the horizontal and vertical lines
  h_coordinates = get_edge_coordinates(bw, True)
  v_coordinates = get_edge_coordinates(bw, False)
  for h in h_coordinates:
    overlay[h, :] = 0
  for v in v_coordinates:
    overlay[:, v] = 0
  #import pdb; pdb.set_trace()
  space_coordinates = np.empty((8,8,4), dtype=int)
  for i in range(8):
    for j in range(8):
      space_coordinates[i, j] = np.array(
        [h_coordinates[i], h_coordinates[i + 1], v_coordinates[j], v_coordinates[j + 1]]
        )
  #return space_coordinates
  cv.imshow("overlay", overlay)
  h_lo, h_hi, v_lo, v_hi = space_coordinates[0, 0]
  show_wait_destroy("a8", src[h_lo:h_hi, v_lo:v_hi])

def get_edge_coordinates(original_image, horizontal):
  axis = 1 if horizontal else 0
  image = np.copy(original_image)
  dim = image.shape[axis]
  size = dim // 30
  # Create structure element for extracting lines through morphology operations
  structure_dim = (size, 1) if horizontal else (1, size)
  structure = cv.getStructuringElement(cv.MORPH_RECT, structure_dim)
  # Apply morphology operations
  image = cv.erode(image, structure)
  image = cv.dilate(image, structure)

  image = cv.bitwise_not(image)
  '''
  Extract edges and smooth image according to the logic
  1. extract edges
  2. dilate(edges)
  3. src.copyTo(smooth)
  4. blur smooth img
  5. smooth.copyTo(src, edges)
  6. detect vertical edges that run the length of the board
  7. Find the 9 edges that are equal distant
  '''
  # Step 1
  edges = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                              cv.THRESH_BINARY, 3, -2)
  # Step 2
  kernel = np.ones((2, 2), np.uint8)
  edges = cv.dilate(edges, kernel)
  # Step 3
  smooth = np.copy(image)
  # Step 4
  smooth = cv.blur(smooth, (2, 2))
  # Step 5
  (rows, cols) = np.where(edges != 0)
  image[rows, cols] = smooth[rows, cols]
  # Step 6
  whole_board_dim = (3, image.shape[1]) if horizontal else (image.shape[0], 3)
  whole_board_kernel = np.ones(whole_board_dim)/np.prod(whole_board_dim)
  image = cv.filter2D(image, -1, whole_board_kernel)
  _, image = cv.threshold(image, 230, 255, cv.THRESH_BINARY)

  # Step 7
  edge_sum = np.sum(image, axis=axis)/ image.shape[axis]
  line_candidates = []
  i = 0
  j = 0
  while i < image.shape[axis]:
    if edge_sum[i] <= 25:
      while i + j < image.shape[axis] and edge_sum[i + j] <= 25:
        j += 1
      line_candidates.append(i + (j - 1)//2)
    i += j + 1
    j = 0

  diffs = np.subtract(line_candidates[1:], line_candidates[:-1])
  filtered_diffs, space_width, max_dist = filter_differences(diffs)

  line_indices = np.where(np.abs(diffs - space_width) <= max_dist)[0]
  coordinates = []
  for i in line_indices:
    coordinates.append(line_candidates[i])
  coordinates.append(int(line_candidates[line_indices[-1]] + filtered_diffs[-1]))
  return coordinates

def filter_differences(diffs):
  #first filter out all the edges that are one pixel
  filtered_diffs = diffs[diffs != 1]
  # for each unique difference value, calculate the average absolute distance 
  # to the 8 nearest diff values
  min_value = None
  min_dist = 1e6

  for candidate in np.unique(filtered_diffs):
    distances = np.abs(filtered_diffs - candidate)
    distances.sort()
    distance = np.mean(distances[:8])
    if distance < min_dist:
      min_value = candidate
      min_dist = distance
      max_dist = np.max(distances[:8])
  filtered_diffs = [d for d in filtered_diffs if np.abs(d - min_value) <= max_dist]
  return filtered_diffs, min_value, max_dist

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

  get_space_coordinates(src)

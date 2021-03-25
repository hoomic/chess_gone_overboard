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

def get_space_coordinates(src, display=False):
  src = cv.resize(src, (300, 300), interpolation=cv.INTER_AREA)
  overlay = np.copy(src)

  if display:
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
  tile_size = src.shape[0] // 10
  clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(tile_size, tile_size))
  gray = clahe.apply(gray)
  if display:
    show_wait_destroy("clahe", gray)

  bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                            cv.THRESH_BINARY, 5, -2)

  if display:
    show_wait_destroy("bw", bw)

  edge_coordinates, h_edges, v_edges = get_board_edges(bw, display)
  # Create the images that will use to extract the horizontal and vertical lines
  h_coordinates = get_grid_coordinates(h_edges, True, edge_coordinates, display)
  v_coordinates = get_grid_coordinates(v_edges, False, edge_coordinates, display)
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
  if display:
    cv.imshow("overlay", overlay)
  h_lo, h_hi, v_lo, v_hi = space_coordinates[0, 0]
  if display:
    show_wait_destroy("a8", src[h_lo:h_hi, v_lo:v_hi])
  return space_coordinates

def get_board_edges(original_image, display=False):
  h_edges = get_edge_coordinates(original_image, True, display)
  v_edges = get_edge_coordinates(original_image, False, display)

  h_flatten = h_edges.mean(axis=0)
  h_diff = np.diff(h_flatten)

  v_flatten = v_edges.mean(axis=1)
  v_diff = np.diff(v_flatten)  

  v_lo = np.argmin(h_diff[:len(h_diff)//3])
  v_hi = np.argmax(h_diff[2 * len(h_diff)//3:]) + 2 * len(v_diff) // 3

  h_lo = np.argmin(v_diff[:len(v_diff)//3])
  h_hi = np.argmax(v_diff[2 * len(v_diff)//3:]) + 2 * len(v_diff) // 3

  return (h_lo-1, h_hi+1, v_lo-1, v_hi+1), h_edges, v_edges

def get_edge_coordinates(original_image, horizontal, edge_coordinates, display=False):
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
  6. detect edges that run the length of the board
  '''
  # Step 1
  edges = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, \
                              cv.THRESH_BINARY, 3, -2)
  # Step 2
  kernel = np.ones((2, 2), np.uint8)
  edges = cv.dilate(edges, kernel)
  if display:
    show_wait_destroy("edges", edges)
  # Step 3
  smooth = np.copy(image)
  # Step 4
  smooth = cv.blur(smooth, (2, 2))
  # Step 5
  (rows, cols) = np.where(edges != 0)
  image[rows, cols] = smooth[rows, cols]
  if display:
    show_wait_destroy("smoothed", image)
  return image

def get_grid_coordinates(image, horizontal, edge_coordinates, display=False):
  axis = 1 if horizontal else 0
  lo = edge_coordinates[0] if horizontal else edge_coordinates[2]
  hi = edge_coordinates[1] if horizontal else edge_coordinates[3]

  # Step 6
  whole_board_dim = (3, image.shape[1]) if horizontal else (image.shape[0], 3)
  whole_board_kernel = np.ones(whole_board_dim)/np.prod(whole_board_dim)
  image = cv.filter2D(image, -1, whole_board_kernel)
  _, image = cv.threshold(image, 230, 255, cv.THRESH_BINARY)
  if display:
    show_wait_destroy("whole_board", image)

  # Step 7
  edge_sum = np.sum(image, axis=axis)/ image.shape[axis]
  line_candidates = []
  i = lo
  j = 0
  while i < hi:
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

  get_space_coordinates(src, argv[1] == "--display")

from pytesseract import Output
import pytesseract
import argparse
import imutils
import cv2


import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import rotate
from skimage.transform import (hough_line, hough_line_peaks)
from scipy.stats import mode
from skimage import io
from skimage.filters import threshold_otsu, sobel
from matplotlib import cm
     
     
     
     
def correct_orientation(image_path):
    # import the necessary packages
    
    # construct the argument parser and parse the arguments
    
    # load the input image, convert it from BGR to RGB channel ordering,
    # and use Tesseract to determine the text orientation
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
    # display the orientation information
    print("[INFO] detected orientation: {}".format(
        results["orientation"]))
    print("[INFO] rotate by {} degrees to correct".format(
        results["rotate"]))
    print("[INFO] detected script: {}".format(results["script"]))
    # rotate the image to correct the orientation
    rotated = imutils.rotate_bound(image, angle=results["rotate"])
    # show the original image and output image after orientation
    # correction
    cv2.imwrite("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/example_answer/orig.jpg", image)
    cv2.imwrite("/home/menglong/workspace/code/multimodal/multimodal_entity_linking/multimodal_entity_linking/output/example_answer/correct.jpg", rotated)
    # cv2.waitKey(0)
    
    
    
def binarizeImage(RGB_image):

  image = rgb2gray(RGB_image)
  threshold = threshold_otsu(image)
  bina_image = image < threshold
  
  return bina_image

def findEdges(bina_image):
  
  image_edges = sobel(bina_image)

  plt.imshow(bina_image, cmap='gray')
  plt.axis('off')
  plt.title('Binary Image Edges')
  plt.savefig('binary_image.png')

  return image_edges

def findTiltAngle(image_edges):
  
  h, theta, d = hough_line(image_edges)
  accum, angles, dists = hough_line_peaks(h, theta, d)
  angle = np.rad2deg(mode(angles)[0][0])
  
  if (angle < 0):
    
    r_angle = angle + 90
    
  else:
    
    r_angle = angle - 90

  # Plot Image and Lines    
  fig, ax = plt.subplots()
  

  ax.imshow(image_edges, cmap='gray')

  origin = np.array((0, image_edges.shape[1]))

  for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):

    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    ax.plot(origin, (y0, y1), '-r')

  ax.set_xlim(origin)
  ax.set_ylim((image_edges.shape[0], 0))
  ax.set_axis_off()
  ax.set_title('Detected lines')

  plt.savefig('hough_lines.png')

  plt.show()
    
  return r_angle

  
def rotateImage(RGB_image, angle):

  fixed_image = rotate(RGB_image, angle)

  plt.imshow(fixed_image)
  plt.axis('off')
  plt.title('Fixed Image')
  plt.savefig('fixed_image.png')
  plt.show()

  return fixed_image

def generalPipeline(img):
 

  image = io.imread(img)
  bina_image = binarizeImage(image)
  image_edges = findEdges(bina_image)
  angle = findTiltAngle(image_edges)
  rotateImage(io.imread(img), angle)
         
    
if __name__ == '__main__':
    generalPipeline("/home/menglong/public_html/for_inner_usage/review_images/0009077-b950014ca6ebb8ad7e0bbaea4a1ff069.jpg")
    # correct_orientation("/home/menglong/public_html/for_inner_usage/review_images/0009077-b950014ca6ebb8ad7e0bbaea4a1ff069.jpg")
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('outputs/blender_rendered/hammer_000_00000027.jpg')

dst = cv2.fastNlMeansDenoisingColored(img,None,7,7,7,21)
cv2.imwrite("test.png", img)
cv2.imwrite("test_smooth.png", dst)
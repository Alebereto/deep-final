from model.MainModel import Painter
import os
import cv2

MODEL_NAME = 'final final'
IMAGE_DIR = 'paint'

#
#   Place images in IMAGE_DIR and run the program.
#   The colored images should be created in the same path with the prefix "colored_".
#

painter = Painter(MODEL_NAME, load=True)

# Get names of files in the image directory
file_names = os.listdir(IMAGE_DIR)

# Get names of images to color
color_names = list()
for name in file_names:
	if (name.endswith('.jpg') or name.endswith('.png')) and not name.startswith('colored_'):
		color_names.append(name)

for name in color_names:
	gray = cv2.imread(os.path.join(IMAGE_DIR, name), cv2.IMREAD_GRAYSCALE)	# read gray image

	color = painter.paint(gray)	# paint gray image

	# save image
	color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
	save_path = os.path.join(IMAGE_DIR, f'colored_{name}')
	cv2.imwrite(save_path, color)
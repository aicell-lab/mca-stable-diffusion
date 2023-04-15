
import pickle
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import random
import colorsys

outdir = "./data/22-fixed"

# Load the pickle file
data = pickle.load(open(os.path.join(outdir, 'predictions.pkl'), 'rb'))
predicted_images = data['prediction']
ref_image = data['reference']
locations = data['locations']


# for each predicted image, we assign a random color map from matplotlib
# then we stack all the color images into one
# and save it


print(len(predicted_images))
# plot the first 20 images in a grid
plt.figure(figsize=(20,12))

# set color map to gray
plt.set_cmap('gray')

for i in range(15):
    plt.subplot(3,5,i+1)
    image = predicted_images[i].sum(axis=2)
    # clip the image to 0-1
    image = np.clip(image, 0, 255) / 255.0
    plt.imshow(image)
    plt.axis('off')
    # plot text in each image with locations
    plt.text(0, 20, locations[i], color='white', fontsize=10)
    
plt.suptitle('Stable Diffusion Predicted Images (condition: location labels, guidance scale = 2.0)')
plt.savefig(os.path.join(outdir, 'predicted-images.png'))



# get all the color maps
cms = list(plt.cm.datad.keys()) # 75 color maps

threshold = 0.7
result_image = np.zeros((predicted_images[0].shape[0], predicted_images[0].shape[1], 3))
# Get the color map by name:
for i in range(len(predicted_images)):
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(255*i) for i in colorsys.hls_to_rgb(h,l,s)]
    image = predicted_images[i].sum(axis=2)
    # clip the image to 0-1
    image = np.clip(image, 0, 255) / 255.0
    # apply threshold and make image binary using scikit image
    
    image[image > threshold] = 1.0
    image[image <= threshold] = 0.0
    
    colored_image = np.stack([image*r, image*g, image*b], axis=2)
    result_image += colored_image
result_image = np.clip(result_image/10, 0, 255)
Image.fromarray(result_image.astype(np.uint8)).save(os.path.join(outdir, 'super-multiplexed.png'))

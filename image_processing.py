import pandas as pd
import numpy as np

from glob import glob

import cv2
import matplotlib.pylab as plt

plt.style.use('ggplot')


cat_files = glob('*.jpg')

#dog_files = glob('../input/cat-and-dog/training_set/training_set/dogs/*.jpg')

img_mpl = plt.imread(cat_files[6])
img_cv2 = cv2.imread(cat_files[6])



###########PIXEL DISTRIBUTION

pd.Series(img_mpl.flatten()).plot(kind='hist',
                                  bins=60,
                                  title='Distribution of Pixel Values')
plt.show()



fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img_mpl)
ax.axis('off')
plt.show()

############################CHANNELS

'''
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img_mpl[:,:,0], cmap='Reds')
axs[1].imshow(img_mpl[:,:,1], cmap='Greens')
axs[2].imshow(img_mpl[:,:,2], cmap='Blues')
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
axs[0].set_title('Red channel')
axs[1].set_title('Green channel')
axs[2].set_title('Blue channel')
plt.show()
'''

'''
#######################COMPARE


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img_cv2)
axs[1].imshow(img_mpl)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set_title('CV2 Image')
axs[1].set_title('Matplotlib Image')
plt.show()


##########################DOG


'''
'''

img_gray = cv2.cvtColor(img_mpl, cv2.COLOR_RGB2GRAY)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_gray, cmap='Greys')
ax.axis('off')
ax.set_title('Grey Image')
plt.show()

'''

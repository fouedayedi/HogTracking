import numpy as np
import cv2
import matplotlib.pyplot as plt
from utile.functions import maxnormgrad,show_gradients,compute_gradients
from skimage.feature import hog
from skimage import io, color
from utile.visuHOG import visuHOG
import pickle



fname = "/home/foued/Desktop/SUPCOM/Tracking/HogTracking/data/test/13.png"

######################
# Section 1 : Question 2, 3
######################
image = cv2.imread(fname)
R, G, B = cv2.split(image)


R_grad_x, R_grad_y, R_grad_mag = compute_gradients(R)
G_grad_x, G_grad_y, G_grad_mag = compute_gradients(G)
B_grad_x, B_grad_y, B_grad_mag = compute_gradients(B) 


show_gradients("Red Channel", R_grad_x, R_grad_y, R_grad_mag)
show_gradients("Green Channel", G_grad_x, G_grad_y, G_grad_mag)
show_gradients("Blue Channel", B_grad_x, B_grad_y, B_grad_mag)


arg_E = (np.arctan2(R_grad_y, R_grad_x) * 180 / np.pi) % 180
arg_G = (np.arctan2(G_grad_y,G_grad_x)* 180 / np.pi ) % 180
arg_B = (np.arctan2(B_grad_y,B_grad_x)* 180 / np.pi ) % 180


magnitudes = np.stack([R_grad_mag, G_grad_mag, B_grad_mag], axis=-1)
angles = np.stack([arg_E, arg_G, arg_B], axis=-1)

A,M = maxnormgrad(magnitudes,angles)

plt.imshow(M, cmap='gray')
plt.colorbar()
plt.title("Maximal Gradient Magnitude (M)")
plt.show()

######################
# Section 2: Questions 4
######################
image = io.imread(fname)
image_gray = color.rgb2gray(image) 

# Compute HOG descriptors with 8*8 pixels_per_cell
fd, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),transform_sqrt=True, cells_per_block=(1, 1), visualize=True) 
visuHOG(image,hog_image) 
# Compute HOG descriptors with 2*2 cells_per_block
fd, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),transform_sqrt=True, cells_per_block=(2, 2), visualize=True)
visuHOG(image,hog_image) 

# Compute HOG descriptors with decalage 
fd, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(3, 3), transform_sqrt=True,cells_per_block=(2, 2), visualize=True)

visuHOG(image,hog_image) 

# Compute HOG descriptors with Norme L2
fd, hog_image = hog(image_gray, orientations=9, pixels_per_cell=(8, 8),transform_sqrt=True, cells_per_block=(2, 2), visualize=True, block_norm='L2')
# Display the HOG image
visuHOG(image,hog_image) 




######################
# Section 3: Questions 5a
######################
# Charger le modèle SVM depuis le fichier classHOG.p
with open('./model/classHOG.p', 'rb') as model_file:
    classifier = pickle.load(model_file)

image_paths = [
    "./INRIAPerson/test_64x128_H96/pos/crop001545a.png",
    "./INRIAPerson/test_64x128_H96/pos/crop001545b.png"
]

for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, block_norm='L2')
    prediction = classifier.predict([fd])
    label = f"Prediction: {prediction[0]}" 
    cv2.imshow(f"Image: {image_path}", image)
    print(f"Prediction for {image_path}: {prediction, label}")

cv2.waitKey(0)
cv2.destroyAllWindows() 

 

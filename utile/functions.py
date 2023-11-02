import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_gradients(channel):
    grad_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    

    
    return grad_x, grad_y, grad_mag 

def maxnormgrad(magnitudes,angles):
    #magnitudes, angles  les cartes des normes et angles sur les 3 canaux couleurs
    #max_norm carte qui donne pour chaque point le canal couleur pour lequel la norme magnitudes 
    #est maximale 
    max_norm = np.argmax(magnitudes, axis=2)

    # Pour chaque pixel, stocker le gradient de norme maximale par rapport aux 3 canaux couleurs
    m, n = np.shape(max_norm)
    I, J = np.ogrid[:m, :n]
    A = angles[I, J, max_norm]
    M = magnitudes[I, J, max_norm]
    return A, M
    
def show_gradients(title, grad_x, grad_y, grad_mag):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(grad_x, cmap='gray')
    plt.title(f"{title} - Gradient X")
    
    plt.subplot(1, 3, 2)
    plt.imshow(grad_y, cmap='gray')
    plt.title(f"{title} - Gradient Y")
    
    plt.subplot(1, 3, 3)
    plt.imshow(grad_mag, cmap='gray')
    plt.title(f"{title} - Magnitude")
    
    plt.tight_layout()
    plt.show()
    






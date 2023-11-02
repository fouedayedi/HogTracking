import matplotlib.pyplot as plt
from skimage import exposure

def visuHOG(I, hog_image, hog_image_scaling_factor=10):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(I, cmap=plt.cm.gray)
    ax1.set_title('Image Originale')

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, hog_image_scaling_factor))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('HOG')
    plt.show()

    return plt

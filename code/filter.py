from PIL import Image # pillow package
import numpy as np
from scipy import ndimage

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr, rescale='minmax'):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.show()

def rgb2gray(arr):
    R = arr[:, :, 0] # red channel
    G = arr[:, :, 1] # green channel
    B = arr[:, :, 2] # blue channel
    gray = 0.2989*R + 0.5870*G + 0.1140*B
    return gray

#########################################
## Please complete following functions ##
#########################################
def sharpen(img, sigma, alpha):
    '''Sharpen the image. 'sigma' is the standard deviation of Gaussian filter. 'alpha' controls how much details to add.'''
    # TODO: Please complete this function.
    # your code here
    
    blurred = ndimage.gaussian_filter(img, sigma=[sigma, sigma, 0])
    detailed = img - blurred
    sharpened = img + alpha * detailed
    # truncated the value
    arr = np.clip(sharpened, 0, 255)
    
    return arr

def median_filter(img, s):
    '''Perform median filter of size s x s to image 'arr', and return the filtered image.'''
    # TODO: Please complete this function.
    # your code here
    
    pad_size = s // 2
    padded_img = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'edge')
    arr = np.zeros_like(img)

    for i in range(pad_size, img.shape[0] + pad_size):
        for j in range(pad_size, img.shape[1] + pad_size):
            for k in range(img.shape[2]):
                neighborhood = padded_img[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1, k]
                arr[i-pad_size, j-pad_size, k] = np.median(neighborhood)
    
    return arr

if __name__ == '__main__':
    input_path = 'E:/1.FilterBasicsAndEdgeDetection/data/rain.jpeg'
    output_path_blur = 'E:/1.FilterBasicsAndEdgeDetection/data/1.1_blur.jpeg'
    output_path_sharpened = 'E:/1.FilterBasicsAndEdgeDetection/data/1.2_sharpened.jpeg'
    output_path_derained = 'E:/1.FilterBasicsAndEdgeDetection/data/1.3_derained.jpeg'
    img = read_img_as_array(input_path)
    # show_array_as_img(img)
    #TODO: finish assignment Part I.
    
    # 1.1
    print(img.shape) # (H,W,C) = (1706, 1279, 3)
    sigma = 2.8  # sigma value
    blurred = ndimage.gaussian_filter(img, sigma=[sigma, sigma, 0])
    save_array_as_img(blurred, output_path_blur)
    
    #1.2
    alpha = 1.5  # alpha value
    sharpened = sharpen(img, sigma, alpha)
    save_array_as_img(sharpened, output_path_sharpened)

    #1.3
    size = 5  # size value for median filter
    median = median_filter(img, size)
    save_array_as_img(median, output_path_derained)

    # Show the results
    show_array_as_img(blurred)
    show_array_as_img(sharpened)
    show_array_as_img(median)

from PIL import Image # pillow package
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

def read_img_as_array(file):
    '''read image and convert it to numpy array with dtype np.float64'''
    img = Image.open(file)
    arr = np.asarray(img, dtype=np.float64)
    return arr

def save_array_as_img(arr, file):
    
    # make sure arr falls in [0,255]
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:
        arr = (arr - min)/(max-min)*255

    arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(file)

def show_array_as_img(arr):
    min, max = arr.min(), arr.max()
    if min < 0 or max > 255:  # make sure arr falls in [0,255]
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

def sobel(arr):
    '''Apply sobel operator on arr and return the result.'''
    # TODO: Please complete this function.
    # your code here
    
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float64)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float64)

    Gx = ndimage.convolve(arr, Kx)
    Gy = ndimage.convolve(arr, Ky)

    G = np.sqrt(np.square(Gx) + np.square(Gy))
    G = np.clip(G, 0, 255)
    return G, Gx, Gy


def nonmax_suppress(G, Gx, Gy):
    '''Suppress non-max value along direction perpendicular to the edge.'''
    assert G.shape == Gx.shape
    assert G.shape == Gy.shape
    row_max, col_max=G.shape
    G_result = G.copy()
    # TODO: Please complete this function.
    # your code here
  
    suppressed_G = np.zeros((row_max,col_max), dtype=np.float64)
    angle = np.arctan2(Gy, Gx) * 180. / np.pi
    angle[angle < 0] = angle[angle < 0] + 180

    for row in range(1, row_max-1):
        for col in range(1, col_max-1):
            try:
                n1 = 255
                n2 = 255

               #angle 0
                if (0 <= angle[row,col] < 22.5) or (157.5 <= angle[row,col] <= 180):
                    n1 = G[row, col+1]
                    n2 = G[row, col-1]
                #angle 45
                elif (22.5 <= angle[row,col] < 67.5):
                    n1 = G[row+1, col-1]
                    n2 = G[row-1, col+1]
                #angle 90
                elif (67.5 <= angle[row,col] < 112.5):
                    n1 = G[row+1, col]
                    n2 = G[row-1, col]
                #angle 135
                elif (112.5 <= angle[row,col] < 157.5):
                    n1 = G[row-1, col-1]
                    n2 = G[row+1, col+1]

                if (G[row,col] >= n1) and (G[row,col] >= n2):
                    suppressed_G[row,col] = G[row,col]
                else:
                    suppressed_G[row,col] = 0

            except IndexError as e:
                pass
                pass
    
    return suppressed_G

def thresholding(G, t):
    '''Binarize G according threshold t'''
    G_binary = G.copy()
    G_binary = (G > t) * 255
    return G_binary

def hysteresis_thresholding(G, low, high):
    '''Binarize G according threshold low and high'''
    # TODO: Please complete this function.
    # your code here
    
    row, col = G.shape
    G_low = thresholding(G, low)
    G_high = thresholding(G, high)
    G_hyst = np.zeros((row,col), dtype=np.int32)

    # Identify strong and weak edges
    strong_i, strong_j = np.where(G >= high)
    zeros_i, zeros_j = np.where(G < low)

    # Set strong edges to max value
    G_hyst[strong_i, strong_j] = 255

    # Perform hysteresis
    weak_i, weak_j = np.where((G >= low) & (G < high))
    
    for i, j in zip(weak_i, weak_j):
        if ((G_hyst[i+1, j-1:j+2] == 255).any() or
            (G_hyst[i-1, j-1:j+2] == 255).any() or
            (G_hyst[i, [j-1, j+1]] == 255).any()):
            G_hyst[i, j] = 255

    G_hyst[zeros_i, zeros_j] = 0
    
    return G_low, G_high, G_hyst

def hough(G_hyst, theta_res=1, p_res=1):
    '''Return Hough transform of G'''
    H, W = G.shape
    D = int(np.ceil(np.sqrt(H ** 2 + W ** 2)))

    # TODO: Please complete this function.
    # your code here
    
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    p = np.linspace(-D, D, int(2 * D / p_res) + 1)  #(p represents rho)
    num_thetas = len(thetas)
    num_p =  len(p)

    votes = np.zeros((num_p, num_thetas), dtype=np.int32)
    
    # Get non-zero points
    y_idxs, x_idxs = np.nonzero(G_hyst)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(num_thetas):
            p_val = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j])) + D)
            if 0 <= p_val < num_p: 
                votes[p_val, j] += 1

    return votes, thetas, p

    # function you may need
    # convert function between theta, p and indices
    def ind2theta(i):
        return i / num_thetas * (2*np.pi)
    def p2ind(p):
        return np.int64(np.ceil(p))
    def ind2p(j):
        return j



    pass

def save_detected_lines(img_path, votes, thetas, ps, save_path, num_peaks=10):
    img = Image.open(img_path)
    output_img = img.convert("RGB")
    draw = ImageDraw.Draw(output_img)

    idxs = np.argpartition(votes.ravel(), -num_peaks)[-num_peaks:]
    peak_p, peak_theta = np.unravel_index(idxs, votes.shape)

    for i in range(len(peak_p)):
        p = ps[peak_p[i]]
        theta = thetas[peak_theta[i]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*p
        y0 = b*p
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)

    output_img.save(save_path)

if __name__ == '__main__':
    input_path = 'E:/1.FilterBasicsAndEdgeDetection/data/road.jpeg'
    save_path_gray = 'E:/1.FilterBasicsAndEdgeDetection/data/2.1_gray.jpeg'
    save_path_Gx = 'E:/1.FilterBasicsAndEdgeDetection/data/2.2_G_x.jpg'
    save_path_Gy = 'E:/1.FilterBasicsAndEdgeDetection/data/2.2_G_y.jpg'
    save_path_G = 'E:/1.FilterBasicsAndEdgeDetection/data/2.2_G.jpg'
    save_path_suppressed_G = 'E:/1.FilterBasicsAndEdgeDetection/data/2.3_supress.jpg'
    
    save_path_low = 'E:/1.FilterBasicsAndEdgeDetection/data/2.4_edgemap_low.jpg'
    save_path_high = 'E:/1.FilterBasicsAndEdgeDetection/data/2.4_edgemap_high.jpg'
    save_path_thresholding = 'E:/1.FilterBasicsAndEdgeDetection/data/2.4_edgemap.jpg'
    
    save_path_hough = 'E:/1.FilterBasicsAndEdgeDetection/data/2.5_hough.jpg'
    save_path_line='E:/1.FilterBasicsAndEdgeDetection/data/2.6_detection_result.jpg'
    
    save_path_line_high='E:/1.FilterBasicsAndEdgeDetection/data/2.7_detection_result_high_resolution.jpg'
    save_path_line_low='E:/1.FilterBasicsAndEdgeDetection/data/2.7_detection_result_low_resolution.jpg'
    
    img = read_img_as_array(input_path)
    #show_array_as_img(img)
    #TODO: finish assignment Part II: detect edges on 'img'
    
    # 2.1
    gray = rgb2gray(img)
    save_array_as_img(gray, save_path_gray)

    # 2.2
    G, Gx, Gy = sobel(gray)
    save_array_as_img(Gx, save_path_Gx)
    save_array_as_img(Gy, save_path_Gy)
    save_array_as_img(G, save_path_G)

    # 2.3
    suppressed_G = nonmax_suppress(G, Gx, Gy)
    save_array_as_img(suppressed_G, save_path_suppressed_G)

    # 2.4
    low_threshold = 50
    high_threshold = 100
    G_low, G_high, G_hyst = hysteresis_thresholding(suppressed_G, low_threshold, high_threshold)
    
    save_array_as_img(G_low, save_path_low)
    save_array_as_img(G_high, save_path_high)
    save_array_as_img(G_hyst, save_path_thresholding)
    
    # 2.5
    H, thetas, p = hough(G_hyst)
    save_array_as_img(H, save_path_hough)
    
    # 2.6
    H_1, thetas_1, p_1 = hough(G_hyst, theta_res=1, p_res=1)
    save_detected_lines(input_path, H, thetas_1, p_1, save_path_line, num_peaks=10)
    
    #2.7
    H_high, thetas_high, p_high = hough(G_hyst, theta_res=0.5, p_res=0.5)
    save_detected_lines(input_path, H_high, thetas_high, p_high, save_path_line_high, num_peaks=10)
    
    H_low, thetas_low, p_low = hough(G_hyst, theta_res=5, p_res=5)
    save_detected_lines(input_path, H_low, thetas_low, p_low, save_path_line_low, num_peaks=10)
    

#writing cell content as .py file for simple import

import numpy as np
import numba

# Using umap's custom metric method to define a hue_saturation_lightness distance metric
@numba.njit()
def hue(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    if delta == 0:
        return 0  # Hue is undefined when delta is zero

    if cmax == r:
        return ((g - b) / delta) % 6
    elif cmax == g:
        return ((b - r) / delta) + 2
    else:
        return ((r - g) / delta) + 4

@numba.njit()
def lightness(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    return (cmax + cmin) / 2.0

@numba.njit()
def saturation(r, g, b):
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    chroma = cmax - cmin
    light = lightness(r, g, b)

    if chroma == 0 or light in [0, 1]:
        return 0  # Saturation is zero when chroma is zero or lightness is at its extremes

    return chroma / (1 - abs(2 * light - 1))

@numba.njit()
def hsl_dist(a, b):
        a_sat = saturation(a[0], a[1], a[2])
        b_sat = saturation(b[0], b[1], b[2])
        a_light = lightness(a[0], a[1], a[2])
        b_light = lightness(b[0], b[1], b[2])
        a_hue = hue(a[0], a[1], a[2])
        b_hue = hue(b[0], b[1], b[2])

        dist = (a_sat - b_sat) ** 2 + (a_light - b_light) ** 2 + (((a_hue - b_hue) % 6) / 6.0) ** 2
    
        return dist
    
@numba.njit()
def hsl_dist_and_grad(a, b):
    dist = hsl_dist(a, b)
    
    # Approximate the gradient using finite differences (very slow)
    epsilon = 1e-5
    grad = np.zeros_like(a)
    for i in range(len(a)):
        a_plus_eps = np.copy(a)
        a_plus_eps[i] += epsilon
        grad[i] = (hsl_dist(a_plus_eps, b) - dist) / epsilon

    return dist, grad

#hsl with a simple sobel edge detection metric added in to see if we can increase cluster separation
@numba.njit()
def sobel_edge_detection(image):
    # Sobel kernels
    Gx = np.array((-1, 0, 1, -2, 0, 2, -1, 0, 1)).reshape((3, 3))
    Gy = np.array((-1, -2, -1, 0, 0, 0, 1, 2, 1)).reshape((3, 3))

    rows, cols = image.shape
    edges = np.zeros((rows, cols))

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Extracting the 3x3 neighborhood
            neighborhood = image[i - 1:i + 2, j - 1:j + 2]

            # Convolution with Sobel kernels
            gx = np.sum(np.multiply(neighborhood, Gx))
            gy = np.sum(np.multiply(neighborhood, Gy))

            # Magnitude of gradients
            edges[i, j] = np.sqrt(gx**2 + gy**2)

    return edges

@numba.njit()
def edge_similarity(img1, img2):
    edges1 = sobel_edge_detection(img1)
    edges2 = sobel_edge_detection(img2)
    return np.sum((edges1 - edges2) ** 2)

@numba.njit()
def grayscale_conversion(image):
    # Manually compute the dot product for grayscale conversion
    rows, cols, _ = image.shape
    grayscale = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            grayscale[i, j] = 0.2989 * image[i, j, 0] + 0.5870 * image[i, j, 1] + 0.1140 * image[i, j, 2]
    return grayscale

@numba.njit()
def hsl_edge_metric(a, b):
    # Reshape flattened images back to 32x32x3
    a_image = a.reshape(32, 32, 3)
    b_image = b.reshape(32, 32, 3)

    # Convert to grayscale for edge detection
    a_gray = grayscale_conversion(a_image)
    b_gray = grayscale_conversion(b_image)

    # Compute distances
    hsl_distance = hsl_dist(a, b)
    edge_distance = edge_similarity(a_gray, b_gray)

    # Combine the distances
    combined = hsl_distance + edge_distance
    
    return combined

@numba.njit()
def ssim_metric(img1, img2, bits_per_pixel=8, k1=0.01, k2=0.03, alpha=1, beta=1, gamma=1):
    # based on the definition provided for the three component SSIM algorithm via wikipedia:
    # URL: https://en.wikipedia.org/wiki/Structural_similarity

    # Constants for SSIM
    L = 2 ** bits_per_pixel - 1
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Calculate means
    mean_x = np.mean(img1)
    mean_y = np.mean(img2)

    # Calculate variances and covariance
    var_x = np.var(img1)
    var_y = np.var(img2)
    cov_xy = np.sum((img1 - mean_x) * (img2 - mean_y)) / (img1.size - 1)

    # Precompute values
    sqrt_var_x = np.sqrt(var_x)
    sqrt_var_y = np.sqrt(var_y)

    # Luminance, contrast, and structure calculations
    luminance = (2 * mean_x * mean_y + C1) / (mean_x**2 + mean_y**2 + C1)
    contrast = (2 * sqrt_var_x * sqrt_var_y + C2) / (var_x + var_y + C2)
    structure = (cov_xy + C2/2) / (sqrt_var_x * sqrt_var_y + C2/2)

    # Combine the three components
    ssim = (luminance ** alpha) * (contrast ** beta) * (structure ** gamma)

    return ssim

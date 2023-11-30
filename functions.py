#!/usr/bin/env python
# coding: utf-8



#functions.ipynb
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import glob
from scipy.signal import find_peaks

def convert(case_value,s):
    switch_dict = {
        'RGB':   cv2.cvtColor(s, cv2.COLOR_BGR2RGB),
        'LUV':  cv2.cvtColor(s, cv2.COLOR_BGR2LUV),
        'HLS':  cv2.cvtColor(s, cv2.COLOR_BGR2HLS),
        'HSV':  cv2.cvtColor(s, cv2.COLOR_BGR2HSV),
        'default':  cv2.cvtColor(s, cv2.COLOR_BGR2RGB),
         }
    return switch_dict.get(case_value, 'Caso non valido')

#load samples for russet detection
def load_samples(folder_path, range_samples,string):
    samples = []
    for i in range(range_samples[0],range_samples[1]):
        
        filename = f"{i}.png"

        image_folder = os.path.join(folder_path, filename)

        if os.path.exists(folder_path) and os.path.exists(image_folder):
            s = cv2.imread(image_folder)
            s = convert(string,s)
            
            s= cv2.resize(s, (20,20))
            samples.append(s)

        else:
            print(f"Sample not found: {folder_path}")
    return samples

#load color and gray images
def load_images(folder_path, range_of_numebrs, string):
    images_color = []
    images_gray = []

    for i in range(range_of_numebrs[0],range_of_numebrs[1]):
        filename_gray = f"C0_00000{i}.png" 
        filename_color = f"C1_00000{i}.png"  
        
        image_path_gray = os.path.join(folder_path, filename_gray)
        image_path_color = os.path.join(folder_path,filename_color )
        if os.path.exists(image_path_gray) and os.path.exists(image_path_color):
            img_gray = cv2.imread(image_path_gray, cv2.IMREAD_GRAYSCALE)
            img_color = cv2.imread(image_path_color)
            img_color = convert(string,img_color)
            images_gray.append(img_gray)
            images_color.append(img_color)

        else:
            print(f"Image not found")

    return images_color,images_gray
    

def resize_images(image, target_size=(20, 20)):
    height, width, _ = image.shape
    resized_image = cv2.resize(image, target_size)
    return resized_images

def show_images(images, titles=None, rows=1, cols=None):

    if titles is not None and len(images) != len(titles):
        raise ValueError("Number of images and titles must match.")

    if cols is None:
        cols = len(images) // rows + (len(images) % rows > 0)

    plt.figure(figsize=(15, 8))  # Adjust the figure size as needed

    for i, image in enumerate(images, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(image, cmap='gray', vmin=0, vmax=255 if len(image) == 2 else None)
        plt.axis('off')

        if titles is not None:
            plt.title(titles[i - 1])

    plt.tight_layout()
    plt.show()

#show histograms using matplotlib
def show_histograms(data_sets, bin_count=10, legend_labels=None):

    plt.figure(figsize=(10, 6))

    for i, data in enumerate(data_sets, 1):
       # hist,bins = np.histogram(d[i].flatten(),256,[0,256])

        plt.hist(data.ravel(), bins=bin_count, alpha=0.5, label=legend_labels[i - 1] if legend_labels else f'Dataset {i}')

    plt.xlabel('Color')
    plt.ylabel('Intensity')
    plt.title('Histograms of Multiple Datasets')
    plt.legend()
    plt.grid(True)
    plt.show()
    

def show_mahalanobis_distance(image, mahalanobis_distances):
    # Visualizza l'immagine originale
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Visualizza la mappa di distanza di Mahalanobis
    plt.subplot(1, 2, 2)
    plt.imshow(mahalanobis_distances)  # Puoi scegliere una mappa colore diversa se preferisci
    
    plt.title('Mahalanobis Distance')
    plt.colorbar()
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def draw_defects(image, element, thickness, scale, min_area, max_area):
    contours = cv2.findContours(element, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours[1]) == 0:
        return
    drawn=0
    for c in contours[1]:
        area = cv2.contourArea(c)
        if min_area < area and len(c) >= 5:
            ellipse = cv2.fitEllipse(c)
            scaled_axes = (ellipse[1][0] * scale, ellipse[1][1] * scale)
            if scaled_axes[0] * scaled_axes[1] * math.pi < max_area:
                scaled_ellipse = ellipse[0], scaled_axes, ellipse[2]
                cv2.ellipse(image, scaled_ellipse, (255, 0, 0), thickness)
                drawn += 1
    return drawn

#linear stetching from course code
def linear_stretching(img, range_percentile):
    hist,bins = np.histogram(img.flatten(),256,[0,256])

    min_value=find_percentile_value(hist, range_percentile[0])
    max_value= find_percentile_value(hist,range_percentile[1])

    
    img[img<min_value] = min_value
    img[img>max_value] = max_value
    
    linear_stretched_img = 255./(max_value-min_value)*(img-min_value)
    return linear_stretched_img


def find_percentile_value(hist, percentile):
    s = 0
    idx = 0
    total_pixel = np.sum(hist)
    while(s < total_pixel*percentile/100):
        s += hist[idx]
        idx += 1
    return idx

def threshold_by_percentile(image, percentile):
    flattened = image.flatten()
    threshold = int(np.percentile(flattened, percentile))
    _, thresholded = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    return thresholded


def find_peaks(histogram):
    peaks = []

    for i in range(2, len(histogram) - 2):
        if np.all((histogram[i] > histogram[i-1]) and (histogram[i] > histogram[i+1]) and (histogram[i-1] > histogram[i-2]) and (histogram[i+1] > histogram[i+2])):
            peaks.append(i)

    return peaks

def find_min_arg(histogram, i1, i2):
    min_arg = np.argmin(histogram[i1:i2+1]) + i1
    return min_arg

def thresh_peaks(img):
    h= cv2.calcHist([img], [0], None, [256], [0, 256])

    peaks = find_peaks(h)

    if len(peaks) >= 2:
        sorted_peaks = sorted(peaks, key=lambda x: h[x], reverse=True)
        i1, i2 = sorted(sorted_peaks[-2:])  # Prendi i due picchi più alti

        return find_min_arg(h, i1, i2)
        

#flood fill using opencv
def flood_fill(binary):
    _,contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros((binary.shape[0]+2, binary.shape[1]+2), np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    mask=255-mask

    for contour in contours:
        seed_point = tuple(contour[0][0])
        cv2.floodFill(binary, mask, seed_point, 255,cv2.FLOODFILL_MASK_ONLY)

    return binary


def fill_back(binary):
    _,contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      # Crea una maschera bianca vuota per floodFill
    mask = np.zeros((binary.shape[0]+2, binary.shape[1]+2), np.uint8)
    largest_contour = max(contours, key=cv2.contourArea)

    # Disegna il contorno del frutto sulla maschera
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=2)

    # Riempire i buchi con flood fill
    for contour in contours:
        seed_point = tuple(contour[0][0])
        cv2.floodFill(binary, mask, seed_point, 0,cv2.FLOODFILL_MASK_ONLY)

    return binary


#delete the background and usless parts from the kiwi
def get_kiwi_from_back(thresholded):
 
    kernel = np.ones((3,3), np.uint8)
    cleaned_image = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)

    _,contours,_ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(thresholded)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    return mask
 
#inspect the fruit and detect the defects
def get_defects(gr, fruit, img_color):
    
    canny = cv2.Canny(gr, 80, 200)
    background =cv2.bitwise_not(fruit)

    kernel = np.ones((5, 5), np.uint8)
    background_dilated = cv2.dilate(background, kernel, iterations=3)
    
    defect = cv2.subtract(canny, background_dilated)
    
    structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    closed = cv2.morphologyEx(defect, cv2.MORPH_CLOSE, structuringElement)

    retval, labels, _, _ = cv2.connectedComponentsWithStats(closed, 4)
    
    display = img_color.copy()

    #_, contours , _= cv2.findContours(fruit, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #cv2.drawContours(display, contours, -1, (0, 0, 255), 1)
    
    return display, retval, labels, closed

def get_russet(fruits, fruit_bw, maha_mask, img_color):
    canny = cv2.Canny(fruits, 80, 200)
    display = img_color.copy()

    russet=fruit_bw.copy()
    _, rus_con , _= cv2.findContours(maha_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(russet, rus_con, -1, 0, thickness=cv2.FILLED)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(russet, cv2.MORPH_CLOSE, structuringElement)

    retval, labels, _, _ = cv2.connectedComponentsWithStats(closed,4)

    _, contours , _= cv2.findContours(fruit_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (0, 0, 255), 1)

    return display, retval, labels



def reference_color(samples):
    means_color=[]

    for i in range(len(samples)):
        means_color.append(np.mean(samples[i], axis=0))
    m=np.mean(means_color, axis=0)
    return np.mean(m, axis=0)

def inv_cov_matrix(samples):
    
    data= np.concatenate(samples, axis=0)
    data = data.reshape(-1, 3)
    cov_matrix = np.cov(data, rowvar=False)

    return cv2.invert(cov_matrix, cv2.DECOMP_SVD)[1]

def normalize_distances(distance):
    
    distance = distance.reshape(image.shape[:2])
    min_distance = np.min(distance)
    max_distance = np.max(distance)
    normalized_distances = (distance - min_distance) / (max_distance - min_distance)*255
    
    return normalized_distances.astype(np.uint8)


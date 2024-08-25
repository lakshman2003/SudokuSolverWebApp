import cv2
import numpy as np
import copy
import math
from scipy import ndimage
from sudoku import *

def WriteOnImage(image, grid, user_grid):
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if(user_grid[i][j] != 0):    
                continue               
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)
            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + off_set_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), font, font_scale, (0,255,0), thickness=2, lineType=cv2.LINE_AA)
    return image

def GetCorners(contours):
    max_iter = 100
    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1
        epsilon = coefficient * cv2.arcLength(contours, True)
        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == 4:
            return hull
        else:
            if len(hull) > 4:
                coefficient += .01
            else:
                coefficient -= .01
    return None

def largest_connected_component(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]
    if(len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image
    max_label = 1
    max_size = sizes[1]     
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(255)
    img2[output == max_label] = 0
    return img2

def reorder(points):
    points = points.reshape((4,2))
    newPoints = np.zeros((4,1,2), dtype=np.int32)
    add = points.sum(1)
    newPoints[0] = points[np.argmin(add)]
    newPoints[2] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    newPoints[1] = points[np.argmin(diff)]
    newPoints[3] = points[np.argmax(diff)]
    result = []
    for i in newPoints:
        result.append(i[0])
    return result

def get_best_shift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)
    rows, cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)
    return shiftx, shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def Solve(img,model):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    biggest = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            biggest = contour
            max_area = area
    
    if biggest is None:
        return img,False
    
    corners = GetCorners(biggest)

    if corners is None:         
        return img,False
    
    corners = reorder(corners)




    pts1 = np.float32(corners)

    (tl, tr, br, bl) = pts1
    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # the height of our Sudoku board
    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    max_width = max(int(width_A), int(width_B))
    max_height = max(int(height_A), int(height_B))


    pts2 = np.float32([[0,0], [max_width-1,0], [max_width-1,max_height-1], [0,max_height-1]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    Warp = cv2.warpPerspective(img, matrix, (max_width,max_height))
    # w = max_width
    # h = max_height
    # a = 15
    Orginal_Warp = np.copy(Warp)
    # Orginal_Warp = Orginal_Warp[a:w-a,a:h-a]
    # cv2.imshow("Orginal Warp",Orginal_Warp)
    # cv2.waitKey(0)
    # Some More Preprocessing
    Warp = cv2.cvtColor(Warp,cv2.COLOR_BGR2GRAY)
    Warp = cv2.GaussianBlur(Warp, (5,5), 0)
    Warp = cv2.adaptiveThreshold(Warp, 255, 1, 1, 11, 2)
    Warp = cv2.bitwise_not(Warp)
    _, Warp = cv2.threshold(Warp, 150, 255, cv2.THRESH_BINARY)

    grid = []
    for r in range(9):
        row = []
        for j in range(9):
            row.append(0)
        grid.append(row)
    

    height = Warp.shape[0] // 9
    width = Warp.shape[1] // 9
    ratio = 0.6
    offset_width = width // 10
    offset_height = height // 10

    for i in range(9):
        for j in range(9):
            #Dividing Sudoku into 9x9:
            crop = Warp[height*i + offset_height : height*(i+1) - offset_height,width*j + offset_width : width*(j+1) - offset_width]
            # Top
            while np.sum(crop[0]) <= (1-ratio) * crop.shape[1] * 255:
                crop = crop[1:]
            # Bottom
            while np.sum(crop[:,-1]) <= (1-ratio) * crop.shape[1] * 255:
                crop = np.delete(crop, -1, 1)
            # Left
            while np.sum(crop[:,0]) <= (1-ratio) * crop.shape[0] * 255:
                crop = np.delete(crop, 0, 1)
            # Right
            while np.sum(crop[-1]) <= (1-ratio) * crop.shape[0] * 255:
                crop = crop[:-1]
            
            crop = cv2.bitwise_not(crop)
            crop = largest_connected_component(crop)
            crop = cv2.resize(crop,(28,28))
            val = 28**2*255 - 28 * 1 * 255
            if crop.sum() >= val:
                grid[i][j] == 0
                continue
            
            center_w = crop.shape[1] // 2
            center_h = crop.shape[0] // 2
            x_start = center_h // 2
            x_end = center_h // 2 + center_h
            y_start = center_w // 2
            y_end = center_w // 2 + center_w
            center_region = crop[x_start:x_end, y_start:y_end]
            
            if center_region.sum() >= center_w * center_h * 255 - 255:
                grid[i][j] = 0
                continue  


            _,crop = cv2.threshold(crop,200,255,cv2.THRESH_BINARY)
            crop = crop.astype(np.uint8)
            crop = cv2.bitwise_not(crop)

            #Centralising the Image
            shift_x, shift_y = get_best_shift(crop)
            shifted = shift(crop,shift_x,shift_y)
            crop = shifted
            crop = cv2.bitwise_not(crop)

            #Making crop Model prediction Ready
            crop = crop.reshape(-1, 28, 28, 1)
            crop = crop.astype('float32')
            crop /= 255

            prediction = model.predict([crop])
            grid[i][j] = np.argmax(prediction[0])+1
    
    userGrid = copy.deepcopy(grid)
    

    print(userGrid)

    
    newBoard = getBoard(grid)
    if(findEmpty(newBoard) is not None):
        return img,False
    if(findEmpty(grid) is None):
        Orginal_Warp = WriteOnImage(Orginal_Warp,newBoard,userGrid)
    
    result = cv2.warpPerspective(Orginal_Warp, matrix, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP)
    result = np.where(result.sum(axis=-1,keepdims=True)!=0, result, img)

    return result,True



    
import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_contour():
    original_image = cv2.imread('.\Dataset_OpenCvDl_Hw2\Q1\coins.jpg')
    image = original_image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(
                gray_blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=30,
                minRadius=20,
                maxRadius=40
            )

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # 複製原始圖像
        img_original = image.copy()

        # 帶有框出圓形輪廓的圖像
        img_with_contours = image.copy()
        for i in circles[0, :]:
            cv2.circle(img_with_contours, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # 只有圓心的圖像
        img_circles_only = np.zeros_like(gray)
        for i in circles[0, :]:
            cv2.circle(img_circles_only, (i[0], i[1]), 2, (255, 255, 255), 3)

        # 顯示原始圖像、帶有圓形輪廓的圖像和只有圓心的圖像
        cv2.imshow('Original Image', img_original)
        cv2.imshow('Circle Contours', img_with_contours)
        cv2.imshow('Circle Centers Only', img_circles_only)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No circles detected.")
        
    return circles
        
def count_coins():
    circles = draw_contour()
    # 顯示計算出的硬幣數量
    num_coins = circles.shape[1]
    output_str = f"There are {num_coins} coins in the image."
    print(output_str)
    
def histogram_equalization_manual(image):
    # 計算直方圖
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])

    # 計算PDF
    pdf = hist / np.sum(hist)

    # 計算CDF
    cdf = np.cumsum(pdf)

    # 創建lookup table
    lookup_table = np.round(cdf * 255).astype('uint8')

    # 應用lookup table
    equalized_image = lookup_table[image]

    # 計算均衡化後的直方圖
    equalized_hist, _ = np.histogram(equalized_image.flatten(), 256, [0, 256])

    return equalized_image, equalized_hist    
    
def histogram_equalization():
    original_image = cv2.imread('.\Dataset_OpenCvDl_Hw2\Q2\histoEqualGray2.png')

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 使用OpenCV進行直方圖均衡化
    equalized_image_opencv = cv2.equalizeHist(original_image)

    # 使用PDF和CDF進行直方圖均衡化
    equalized_image_manual, equalized_hist_manual = histogram_equalization_manual(original_image)

    # 顯示原始圖像、OpenCV均衡化圖像和手動均衡化圖像
    plt.subplot(2, 3, 1), plt.imshow(original_image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 2), plt.imshow(equalized_image_opencv, cmap='gray')
    plt.title('Equalized with OpenCV'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 3, 3), plt.imshow(equalized_image_manual, cmap='gray')
    plt.title('Equalized Manually'), plt.xticks([]), plt.yticks([])

    # 顯示原始圖像和均衡化圖像的直方圖
    plt.subplot(2, 3, 4)
    plt.bar(range(256), cv2.calcHist([original_image], [0], None, [256], [0, 256]).flatten(), color='blue', alpha=0.7)
    plt.title('Histogram of Original'), plt.xlim([0, 256])

    plt.subplot(2, 3, 5)
    plt.bar(range(256), cv2.calcHist([equalized_image_opencv], [0], None, [256], [0, 256]).flatten(), color='blue', alpha=0.7)
    plt.title('Histogram of Equalized (OpenCV)'), plt.xlim([0, 256])

    plt.subplot(2, 3, 6)
    plt.bar(range(256), equalized_hist_manual, color='blue', alpha=0.7)
    plt.title('Histogram of Equalized (Manual)'), plt.xlim([0, 256])

    plt.show()
    
def closing_operation(image, kernel_size):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Pad the image with zeros based on the kernel size
    padded_image = np.pad(binary_image, pad_width=kernel_size // 2, mode='constant')

    # Create a 3x3 all-ones structuring element
    structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Perform dilation operation
    dilated_image = np.zeros_like(padded_image)
    for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
            region = padded_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, 
                                  j - kernel_size // 2 : j + kernel_size // 2 + 1]
            dilated_image[i, j] = np.max(region)

    # Perform erosion operation
    eroded_image = np.zeros_like(padded_image)
    for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
            region = padded_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, 
                                  j - kernel_size // 2 : j + kernel_size // 2 + 1]
            eroded_image[i, j] = np.min(region)

    # Show the result
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 4, 1), plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image (Grayscale)'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 2), plt.imshow(binary_image, cmap='gray')
    plt.title('Binarized Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 3), plt.imshow(dilated_image, cmap='gray')
    plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 4), plt.imshow(eroded_image, cmap='gray')
    plt.title('Eroded Image'), plt.xticks([]), plt.yticks([])

    plt.show()
    
def opening_operation(image, kernel_size):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarize the grayscale image
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Pad the image with zeros based on the kernel size
    padded_image = np.pad(binary_image, pad_width=kernel_size // 2, mode='constant')

    # Create a 3x3 all-ones structuring element
    structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    # Perform erosion operation
    eroded_image = np.zeros_like(padded_image)
    for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
            region = padded_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, 
                                  j - kernel_size // 2 : j + kernel_size // 2 + 1]
            eroded_image[i, j] = np.min(region)

    # Perform dilation operation
    dilated_image = np.zeros_like(padded_image)
    for i in range(kernel_size // 2, padded_image.shape[0] - kernel_size // 2):
        for j in range(kernel_size // 2, padded_image.shape[1] - kernel_size // 2):
            region = padded_image[i - kernel_size // 2 : i + kernel_size // 2 + 1, 
                                  j - kernel_size // 2 : j + kernel_size // 2 + 1]
            dilated_image[i, j] = np.max(region)

    # Show the result
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 4, 1), plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image (Grayscale)'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 2), plt.imshow(binary_image, cmap='gray')
    plt.title('Binarized Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 3), plt.imshow(eroded_image, cmap='gray')
    plt.title('Eroded Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(1, 4, 4), plt.imshow(dilated_image, cmap='gray')
    plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])

    plt.show()
    
def morphology_operation_closing():
    rgb_image = cv2.imread('.\Dataset_OpenCvDl_Hw2\Q3\closing.png')
    
    # 指定結構元素的大小（3x3）
    kernel_size = 3

    # 執行運算
    closing_operation(rgb_image, kernel_size)
    
def morphology_operation_opening():
    rgb_image = cv2.imread('.\Dataset_OpenCvDl_Hw2\Q3\opening.png')
    
    # 指定結構元素的大小（3x3）
    kernel_size = 3

    # 執行運算
    opening_operation(rgb_image, kernel_size)


def main():
    while True:
        select = int(input('Choose question number: '))
        if select == 1:        
            draw_contour()
            count_coins()
        elif select == 2:
            histogram_equalization()
        elif select == 3:
            morphology_operation_closing()
            morphology_operation_opening()
        elif select == 0:
            break

if __name__ == "__main__":
    main()
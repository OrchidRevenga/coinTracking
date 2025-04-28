import cv2
import numpy as np
import matplotlib.pyplot as plt

def coinTracking(image_path, show_result=True):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    blurred = cv2.GaussianBlur(gray, (3, 5), 0)
    cv2.imshow("blur", blurred)

    edged = cv2.Canny(blurred, 100, 200)
    cv2.imshow("edge", edged)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tray_contour = max(contours, key=cv2.contourArea)
    tray_area = cv2.contourArea(tray_contour)

    for n in contours:
        print(cv2.contourArea(n))

    tray_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(tray_mask, [tray_contour], -1, 255, -1)
    cv2.imshow("Mask", tray_mask)

    coins = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                             param1=70, param2=50, minRadius=20, maxRadius=40)

    big_inside = small_inside = big_outside = small_outside = 0

    output = image.copy()
    cv2.drawContours(output, [tray_contour], -1, (255, 0, 0), 2)

    if coins is not None:
        coins = np.round(coins[0, :]).astype("int")
        print(coins)
        for (x, y, r) in coins:
            inside_tray = tray_mask[y, x] > 0
            big = r > 31

            if inside_tray:
                if big:
                    big_inside += 1
                else:
                    small_inside += 1
            else:
                if big:
                    big_outside += 1
                else:
                    small_outside += 1

            color = (0, 0, 255) if big else (0, 255, 0)
            cv2.circle(output, (x, y), r, color, 2)
            cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)

    if show_result:
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("Detected")
        plt.axis("off")
        plt.show()

    return {
        "tray_area": tray_area,
        "big_inside": big_inside,
        "small_inside": small_inside,
        "big_outside": big_outside,
        "small_outside": small_outside
    }

if __name__ == "__main__":
    result = coinTracking("tray6.jpg")
    for key, value in result.items():
        print(f"{key}: {value}")

import cv2
import numpy as np
import matplotlib as plt


def get_roi(image, roi_coords):
    roi_coords = np.array([roi_coords], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, roi_coords, 255)
    roi_segment = cv2.bitwise_and(image, mask)
    return roi_segment


def auto_canny_edge_detector(image, sigma=0.25):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(image, lower, upper)
    return edges


def blend_images(image, final_image, alpha=0.7, beta=1., gamma=0.):
    return cv2.addWeighted(final_image, alpha, image, beta, gamma)


def main():
    color_dict_HSV = {
        'black': [[180, 255, 30], [0, 0, 0]],
        'white': [[180, 18, 255], [0, 0, 231]],
        'red1': [[180, 255, 255], [159, 50, 70]],
        'red2': [[9, 255, 255], [0, 50, 70]],
        'green': [[89, 255, 255], [36, 50, 70]],
        'blue': [[128, 255, 255], [90, 50, 70]],
        'yellow': [[35, 255, 255], [25, 50, 70]],
        'purple': [[158, 255, 255], [129, 50, 70]],
        'orange': [[24, 255, 255], [10, 50, 70]],
        'gray': [[180, 18, 230], [0, 0, 40]]
    }

    image_path = r'/home/jake/cv_area_mapping/images/EMPTY_FRAME_MULTIPLE_SHOOTER_OVERCAST_ZOOMED_OUT_GX010003.png'

    # Load a frame (or image)
    image = cv2.imread(image_path)
    image = cv2.resize(image, None, fx=.2, fy=.2, interpolation=cv2.INTER_AREA)
    y_size, x_size = image.shape[:2]

    # hardcode vertices for ROI initialy, top_left then counter clockwise
    roi_coords = [
        (0, 0),
        (354, 180),
        (156, 256),
        (0, 312),
        (0, 431),
        (767, 431),
        (767, 311),
        (412, 175),
        (767, 0)
    ]

    denoise_2 = cv2.fastNlMeansDenoisingColored(image.copy(), None, 5, 5, 7, 21)
    gray_image = cv2.cvtColor(denoise_2.copy(), cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray_image.copy(), (5, 5), 0)
    canny_image = auto_canny_edge_detector(gray_blur.copy())
    erode_canny = cv2.erode(canny_image.copy(), (3, 3), iterations=3)
    dialated_canny = cv2.dilate(erode_canny.copy(), (21, 21), iterations=3)
    roi_frame = get_roi(canny_image.copy(), roi_coords)

    # Hough Lines params
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 50
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 40  # maximum gap in pixels between connectable line segments

    lines = cv2.HoughLinesP(roi_frame, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    # convert each line to coordinates back in the original image
    output = np.zeros_like(image)  # creating a blank to draw lines on
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)

    orginalImageWithHoughLines = blend_images(output, image)
    cv2.imshow('image', image)
    cv2.imshow('canny_image', canny_image)
    cv2.imshow('output', output)
    cv2.imshow('roi_frame', roi_frame)
    cv2.imshow('orginalImageWithHoughLines', orginalImageWithHoughLines)
    cv2.waitKey()


if __name__ == "__main__":
    main()

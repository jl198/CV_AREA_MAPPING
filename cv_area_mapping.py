from __future__ import division

import cv2
import numpy as np
import track


class LaneDetector:
    def __init__(self, road_horizon, prob_hough=True):
        self.prob_hough = prob_hough
        self.vote = 50
        self.roi_theta = 0.2
        self.road_horizon = road_horizon

    @staticmethod
    def _standard_hough(img, init_vote):
        """Hough transform wrapper to return a list of points like PHough does
        """
        lines = cv2.HoughLines(img, 1, np.pi / 180, init_vote)
        points = [[]]
        for l in lines:
            for rho, theta in l:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                points[0].append((x1, y1, x2, y2))
        return points

    @staticmethod
    def _base_distance(x1, y1, x2, y2, width):
        """Compute the point where the give line crosses the base of the frame
        and return distance of that point from center of the frame
        """
        if x2 == x1:
            return (width * 0.5) - x1
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        base_cross = -c / m
        return (width * 0.5) - base_cross

    def _scale_line(self, x1, y1, x2, y2, frame_height):
        """Scale the farthest point of the segment to be on the drawing horizon
        """
        #
        if x1 == x2:
            if y1 < y2:
                y1 = self.road_horizon
                y2 = frame_height
                return x1, y1, x2, y2
            else:
                y2 = self.road_horizon
                y1 = frame_height
                return x1, y1, x2, y2
        if y1 < y2:
            m = (y1 - y2) / (x1 - x2)
            x1 = ((self.road_horizon - y1) / m) + x1
            y1 = self.road_horizon
            x2 = ((frame_height - y2) / m) + x2
            y2 = frame_height
        else:
            m = (y2 - y1) / (x2 - x1)
            x2 = ((self.road_horizon - y2) / m) + x2
            y2 = self.road_horizon
            x1 = ((frame_height - y1) / m) + x1
            y1 = frame_height
        return x1, y1, x2, y2

    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roiy_end = frame.shape[0] # 432
        roix_end = frame.shape[1] # 768
        roi = img[self.road_horizon:roiy_end, 0:roix_end] # img[256:432, 0:768] -> img[startY:endY, startX:endX]
        # print(f'({roix_end=},{roiy_end=}):\t') # roi (y1,y2):{roi[0]}, (x1,x2):{roi[1]}
        blur = cv2.medianBlur(roi, 5)
        contours = cv2.Canny(blur, 60, 120)
        # print(contours is not None)
        if self.prob_hough:
            lines = cv2.HoughLinesP(contours, 1, np.pi / 180, self.vote,
                                    minLineLength=30, maxLineGap=100)
        else:
            lines = self._standard_hough(contours, self.vote)

        if lines is not None:
            # find nearest lines to center
            # scale points from ROI coordinates to full frame coordinates
            lines = lines + np.array([0, self.road_horizon,
                                      0, self.road_horizon]).reshape((1, 1, 4))
            left_bound = None
            right_bound = None
            for l in lines:
                # find the rightmost/leftmost line of the left/right half
                for x1, y1, x2, y2 in l:
                    # line angle WRT horizon
                    theta = np.abs(np.arctan2((y2 - y1), (x2 - x1)))
                    if theta > self.roi_theta:  # ignore lines with small angle
                        dist = self._base_distance(x1, y1, x2, y2,
                                                   frame.shape[1])
                        if left_bound is None and dist < 0:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is None and dist > 0:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
                        elif left_bound is not None and 0 > dist > left_dist:
                            left_bound = (x1, y1, x2, y2)
                            left_dist = dist
                        elif right_bound is not None and 0 < dist < right_dist:
                            right_bound = (x1, y1, x2, y2)
                            right_dist = dist
            if left_bound is not None:
                left_bound = self._scale_line(left_bound[0], left_bound[1],
                                              left_bound[2], left_bound[3],
                                              frame.shape[0])
            if right_bound is not None:
                right_bound = self._scale_line(right_bound[0], right_bound[1],
                                               right_bound[2], right_bound[3],
                                               frame.shape[0])

            return roi, [roix_end, roiy_end], [left_bound, right_bound]


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--path", required=True, type=str, help="Video path")
#     return parser.parse_args()


def main(video_path):
    cap = cv2.VideoCapture(video_path)

    ticks = 0
    try:
        lt = track.LaneTracker(2, 0.1, 500)
    except ValueError:
        print("OH NO FUCK ME")
    ld = LaneDetector(256)
    while cap.isOpened():
        precTick = ticks
        ticks = cv2.getTickCount()
        dt = (ticks - precTick) / cv2.getTickFrequency()

        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.resize(frame, None, fx=.2, fy=.2, interpolation=cv2.INTER_AREA)
        # frame = np.array([frame], np.uint32)
        predicted = lt.predict(dt)
        print(f'{predicted=}')

        roi, roi_cord_ends, lanes = ld.detect(frame)
        # list = roi
        print(frame)
        if predicted is not None:
            cv2.line(frame,
                     (predicted[0][0], predicted[0][1]),
                     (predicted[0][2], predicted[0][3]),
                     (0, 0, 255), 5)
            cv2.line(frame,
                     (predicted[1][0], predicted[1][1]),
                     (predicted[1][2], predicted[1][3]),
                     (0, 0, 255), 5)

        if lanes is not None:
            lt.update(lanes)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    # args = parse_args()
    # main(args.path)
    video_path = r'/home/jake/cv_area_mapping/videos/MULTIPLE_SHOOTER_OVERCAST_ZOOMED_OUT_GX010003.MP4'
    main(video_path)

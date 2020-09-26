import cv2
import numpy as np


class Parameters:

    def __init__(self):
        self.font_size = .6
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.scaleFactor = 0.4
        self.scaleUp = 3 / 4
        self.thresh = 15
        self.width = 640
        self.height = 480
        self.tol = 50
        self.neutralBool = False
        self.motor_thresh = int(50 * self.scaleFactor)
        self.pos_lower = (np.arange(175, 450, 25) * self.scaleUp).astype(int)
        self.pos_upper = (np.arange(25, 175, 25) * self.scaleUp).astype(int)
        self.pos_emotion = (np.arange(25, 225, 25) * self.scaleUp).astype(int)
        self.centerFixed = np.array((int(self.width * self.scaleFactor / 2), int(self.height * self.scaleFactor / 2)))
        self.yCenter = self.centerFixed[1]
        self.xCenter = self.centerFixed[0]

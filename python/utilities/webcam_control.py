#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:59:59 2020

@author: mup
"""

import cv2
import os
import numpy as np
from skimage.transform import downscale_local_mean

N_PARAMS = 42  # Number of parameters in settings file


class Webcam():
    """
        Class to read from a video stream (camera or video file)

        Args:
            port (int or string): Video port or file to open
            settings (dict or string)): Camera settings
            downscale (float): downscaling factor for streamed images
    """

    def __init__(self, port=0, settings=None, downscale=1):
        super(Webcam, self).__init__()

        self._stop = False
        self.scale = (downscale,)*2 + (1,)
        if settings is None:
            settings = {'frame_width': 2048, 'frame_height': 1536,
                        'exposure': -4, 'gain': 0}

        # Open the device at location 'port'
        print('Try to open camera...')
        self.cap = cv2.VideoCapture(port)

        # Check whether user selected video stream opened successfully.
        if not (self.cap.isOpened()):
            raise IOError("Could not open camera at port {}".format(port))
        print('Camera opened successfully')

        # Camera setting
        print('Write settings...')
        if type(settings) is str:
            # Read settings from file
            with open(settings, "r") as f:
                content = f.read()
                props = content.split(',')
                props = np.array(props, dtype='float')
                for i in range(N_PARAMS):
                    self.cap.set(i, props[i])
        else:
            # Read settings from dictionary
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['frame_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['frame_height'])
            self.cap.set(cv2.CAP_PROP_GAIN, settings['gain'])
            self.cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
        print('Camera is initialized')

    def stop_camera(self, arg=None):
        """ Stop streaming

        Args:
            arg (object): Dummy parameter to use as event callback
        """
        self._stop = True

    def enable_frames(self):
        """ Enable to take frames with camera
        """
        self._stop = False

    def get_frame(self):
        """ Get next frame (image) from stream
        """
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27 or self._stop:
            return False, 0
        ret, frame = self.cap.read()
        frame = np.round(
            downscale_local_mean(frame, self.scale)).astype('uint8')
        return ret, frame

    def close(self,):
        """ Close connection to video stream
        """
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    from time import time
    # from scipy.ndimage.filters import correlate
    from scipy.signal import fftconvolve as correlate
    from joblib import Parallel
    camera = Webcam(0, downscale=1, settings=None)
    print('Start streaming...')
    alpha = 0.8
    fps = 0
    min_win_size = 3
    max_win_size = 23
    win_step = 10
    C = 7
    n_jobs = (max_win_size - min_win_size)//win_step + 1
    mean_filters = []
    for i in range(n_jobs):
        N = min_win_size + i*win_step
        mean_filters.append(np.ones((N, N))/N**2)
    # aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)

    while 1:
        t = time()
        ret, frame = camera.get_frame()
        # cv2.imshow('Image', frame)
        frame = cv2.putText(frame, f'{fps:.1f} FPS', (20, 450), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255))
        f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # thresh_frame = cv2.adaptiveThreshold(f, 255,
        #                                       cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                       cv2.THRESH_BINARY_INV, 23, 7)

        # Parallel(n_jobs)()
        thresh_frame = (255*(f < (correlate(f, mean_filters, 'same')-C))).astype(np.uint8)
        # contours, _ = cv2.findContours(thresh_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, aruco_dict)
        # frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        cv2.imshow('Image', frame)
        cv2.imshow('Thresholded', thresh_frame.astype(np.uint8))
        # cv2.imshow('Boundaries', cv2.drawContours(
        #     thresh_frame.astype(np.uint8), contours, -1, (0, 0, 255), 3))
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        t2 = time()
        fps = alpha*fps + (1-alpha)/(t2-t)
        t = t2
    camera.close()
    print('Closed camera')

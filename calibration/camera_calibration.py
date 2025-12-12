#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

# === user setting ===
CHESSBOARD_SIZE = (10, 7)    # corners (columns, rows)
SQUARE_SIZE = 0.15         # length in m
CAPTURE_MIN_VIEWS = 15      # minimum number of images to perform calibration
CAMERA_INDEX = 0            # camera index

SAVE_FILE = "calib_chessboard.npz"

def main():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = []   # 3D points (world frame)
    imgpoints = []   # 2D points (camera frame)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("can't open camera in index =", CAMERA_INDEX)
        return

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    gray = None

    print("[INFO] press SPACE to capture one image and q to stop calibration。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] can't acquire frame。")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

        if found:
            corners_sub = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )
            cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners_sub, found)
            cv2.putText(frame, "SPACE=Save  q=Quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No chessboard detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Chessboard Capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE
            if found:
                objpoints.append(objp.copy())
                imgpoints.append(corners_sub)
                print(f"[INFO] collected {len(objpoints)} frames")
            else:
                print("[INFO] no chessboard detected")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < CAPTURE_MIN_VIEWS:
        print(f"[ERROR] not enough value ({len(objpoints)}/{CAPTURE_MIN_VIEWS}) can't perform calibration。")
        return

    # === Calibration ===
    h, w = gray.shape[:2]
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    print("\n=== Chessboard Calibration Result ===")
    print("RMS reprojection error:", ret)
    print("cameraMatrix:\n", cameraMatrix)
    print("distCoeffs:\n", distCoeffs.ravel())

    # storing data
    np.savez(
        SAVE_FILE,
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        rms=ret,
        image_width=w,
        image_height=h
    )
    print(f"[INFO] Calibration parater stored {os.path.abspath(SAVE_FILE)}")

    # === Undistort preview ===
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[WARN] skip undistort preview")
        return

    print("[INFO] Undistort previewing, press q to exit。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix, distCoeffs, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)

        cv2.imshow("Original", frame)
        cv2.imshow("Undistorted", undistorted)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

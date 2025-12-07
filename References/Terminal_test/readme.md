Test 2D lidar-Camera sensor fusion

5 terminals are used

1. roscore

2. roslaunch usb_cam usb_cam-test.launch

3. roslaunch rplidar_ros rplidar_a1.launch

4. rosrun lidar lidar_projection.py

5. rqt_image_view

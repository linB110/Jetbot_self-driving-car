#pragma once

#include <memory>
#include <vector>
#include <string>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>

#include "trt_yolo.h"
#include "utils.h"
#include <yolo_detection/detection.h>

class YoloRosNode
{
public:
    explicit YoloRosNode(ros::NodeHandle& nh);
    ~YoloRosNode();

    bool isOk() const { return ok_; }

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;
    ros::Publisher detPub_;

    std::unique_ptr<TrtYolo> yolo_;
    bool ok_;
    bool showImage_;
    int  detectEveryNFrames_;

    int frameIdx_;
    std::vector<BBox> lastBoxes_;
    float lastInferTime_;
    ros::Time lastDispTime_;
};

#include <ros/ros.h>
#include <cuda_runtime_api.h>
#include "yolo_ros_node.h"

int main(int argc, char** argv)
{
    // establish GPU context
    cudaFree(0);
    
    // init ros
    ros::init(argc, argv, "yolo_detection");
    ros::NodeHandle nh;

    YoloRosNode node(nh);
    if (!node.isOk())
    {
        ROS_ERROR("YoloRosNode initialization failed");
        return 1;
    }

    ros::spin();
    return 0;
}

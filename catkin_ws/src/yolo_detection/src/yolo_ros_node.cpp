#include "yolo_ros_node.h"

#include <XmlRpcValue.h>

YoloRosNode::YoloRosNode(ros::NodeHandle& nh)
    : nh_(nh),
      it_(nh_),
      yolo_(nullptr),
      ok_(false),
      showImage_(false),
      detectEveryNFrames_(5),
      frameIdx_(0),
      lastInferTime_(0.0f)
{
    ros::NodeHandle pnh("~");

    // 1. read parameter from yaml file
    YoloConfig cfg;

    pnh.param<std::string>("engine_path", cfg.enginePath,
                           std::string("/home/lab605/yolo/yolov4.trt"));
    pnh.param("input_w", cfg.inputW, 320);
    pnh.param("input_h", cfg.inputH, 320);
    pnh.param("input_c", cfg.inputC, 3);

    pnh.param("conf_thresh", cfg.confThresh, 0.8f);
    pnh.param("iou_thresh", cfg.iouThresh, 0.3f);
    pnh.param("detect_every_n_frames", detectEveryNFrames_, 5);
    pnh.param("show_image", showImage_, false);

    // class_names (YAML list -> std::vector<std::string>)
    XmlRpc::XmlRpcValue cls_list;
    if (pnh.getParam("class_names", cls_list) &&
        cls_list.getType() == XmlRpc::XmlRpcValue::TypeArray)
    {
        for (int i = 0; i < cls_list.size(); ++i)
        {
            if (cls_list[i].getType() == XmlRpc::XmlRpcValue::TypeString)
            {
                std::string name = static_cast<std::string>(cls_list[i]);
                cfg.classNames.push_back(name);
            }
        }
    }
    else
    {
        ROS_WARN("no class_names param, use default list");
        cfg.classNames = {
            "Ahead-only", "General-caution", "No-entry", "Pedestrians",
            "Speed Limit 100", "Speed Limit 120", "Speed Limit 20",
            "Speed Limit 30", "Speed Limit 50", "Speed Limit 60",
            "Speed Limit 70", "Speed Limit 80", "Stop",
            "Turn-left-ahead", "Turn-right-ahead"
        };
    }

    ROS_INFO_STREAM("YOLO engine_path = " << cfg.enginePath);
    ROS_INFO("YOLO input (W,H,C) = (%d, %d, %d)", cfg.inputW, cfg.inputH, cfg.inputC);
    ROS_INFO("YOLO conf_thresh = %.3f, iou_thresh = %.3f",
             cfg.confThresh, cfg.iouThresh);
    ROS_INFO("YOLO detect_every_n_frames = %d", detectEveryNFrames_);
    ROS_INFO("YOLO show_image = %s", showImage_ ? "true" : "false");

    // 2. initialize TensorRT YOLO
    yolo_.reset(new TrtYolo(cfg));
    if (!yolo_ || !yolo_->isOk())
    {
        ROS_ERROR("TRT initialization failed，check engine and params");
        ok_ = false;
        return;
    }

    // 3. ROS I/O
    sub_ = it_.subscribe("/usb_cam/image_raw", 1,
                         &YoloRosNode::imageCallback, this);

    detPub_ = nh_.advertise<yolo_detection::detection>("/yolo/detection_node", 10);

    lastDispTime_ = ros::Time::now();

    if (showImage_)
    {
        cv::namedWindow("YOLOv4 TRT Detection", cv::WINDOW_AUTOSIZE);
    }

    ok_ = true;
    ROS_INFO("YOLOv4 TensorRT C++ node (subscribe /usb_cam/image_raw) has activated");
}

YoloRosNode::~YoloRosNode()
{
    if (showImage_)
    {
        cv::destroyAllWindows();
    }
}

void YoloRosNode::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    if (!ok_ || !yolo_)
        return;

    cv_bridge::CvImageConstPtr cvPtr;
    try
    {
        cvPtr = cv_bridge::toCvShare(msg, "bgr8");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge transform failed: %s", e.what());
        return;
    }

    const cv::Mat& frame = cvPtr->image;
    if (frame.empty())
    {
        ROS_WARN("receive empty image");
        return;
    }

    frameIdx_++;

    // display FPS
    ros::Time now = ros::Time::now();
    double dt = (now - lastDispTime_).toSec();
    lastDispTime_ = now;
    double dispFps = (dt > 1e-6) ? (1.0 / dt) : 0.0;

    cv::Mat showImg = frame.clone();

    // inference evey N frames
    if (frameIdx_ % detectEveryNFrames_ == 0)
    {
        lastBoxes_.clear();
        if (yolo_->infer(frame, lastBoxes_, showImg, lastInferTime_))
        {
            const auto& classNames = yolo_->classNames();

            // publish detection result
            for (const auto& b : lastBoxes_)
            {
                yolo_detection::detection msgOut;

                if (b.cls_id >= 0 &&
                    b.cls_id < static_cast<int>(classNames.size()))
                {
                    msgOut.class_name = classNames[b.cls_id];
                }
                else
                {
                    msgOut.class_name = "unknown";
                }

                float cx = (b.x1 + b.x2) * 0.5f;
                float cy = (b.y1 + b.y2) * 0.5f;
                msgOut.cx = cx;
                msgOut.cy = cy;

                detPub_.publish(msgOut);
            }
        }
        else
        {
            ROS_WARN("YOLO inference failed");
        }
    }
    else
    {
        // no inference -> draw last detection
        const auto& classNames = yolo_->classNames();
        for (const auto& b : lastBoxes_)
        {
            int x1 = static_cast<int>(b.x1);
            int y1 = static_cast<int>(b.y1);
            int x2 = static_cast<int>(b.x2);
            int y2 = static_cast<int>(b.y2);

            std::string label = "id_" + std::to_string(b.cls_id);
            if (b.cls_id >= 0 &&
                b.cls_id < static_cast<int>(classNames.size()))
            {
                label = classNames[b.cls_id];
            }

            cv::rectangle(showImg, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 2);

            char text[256];
            std::snprintf(text, sizeof(text), "%s", label.c_str());
            cv::putText(showImg, text, cv::Point(x1, std::max(0, y1 - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }
    }

    // show FPS
    double inferFps = (lastInferTime_ > 1e-6) ? (1.0 / lastInferTime_) : 0.0;
    char fpsText[256];

    std::snprintf(fpsText, sizeof(fpsText), "Disp FPS: %.2f", dispFps);
    cv::putText(showImg, fpsText, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);

    std::snprintf(fpsText, sizeof(fpsText),
                  "Infer FPS: %.2f (every %d frames)",
                  inferFps, detectEveryNFrames_);
    cv::putText(showImg, fpsText, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);

    if (showImage_)
    {
        cv::Mat disp;
        cv::resize(showImg, disp, cv::Size(320, 320));
        cv::imshow("YOLOv4 TRT Detection", disp);
        int key = cv::waitKey(1);
        if ((key & 0xFF) == 'q')
        {
            cv::destroyWindow("YOLOv4 TRT Detection");
        }
    }
}

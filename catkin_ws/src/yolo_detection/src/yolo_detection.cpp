#include <ros/ros.h>
#include <yolo_detection/detection.h>

#include <opencv2/opencv.hpp>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// ================== parameters ==================

static const std::string ENGINE_PATH = "/home/lab605/yolo/yolov4.trt";

// inference size (should be identical as training size)
static const int INPUT_H = 320;
static const int INPUT_W = 320;
static const int INPUT_C = 3;

// class names
static const std::vector<std::string> CLASS_NAMES = {
    "Ahead-only", "General-caution", "No-entry", "Pedestrians",
    "Speed Limit 100", "Speed Limit 120", "Speed Limit 20",
    "Speed Limit 30", "Speed Limit 50", "Speed Limit 60",
    "Speed Limit 70", "Speed Limit 80", "Stop",
    "Turn-left-ahead", "Turn-right-ahead"
};

static const int NUM_CLASSES   = CLASS_NAMES.size();
static const float CONF_THRESH = 0.8f;
static const float IOU_THRESH  = 0.3f;
static const int   DETECT_EVERY_N_FRAMES = 5;

// ================== TensorRT Logger ==================

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

// ================== helpers：IoU / NMS ==================

struct BBox
{
    float x1, y1, x2, y2;
    float score;
    int   cls_id;
};

static float bboxIoU(const BBox& a, const BBox& b)
{
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    if (inter <= 0.0f) return 0.0f;

    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni = areaA + areaB - inter;
    if (uni <= 0.0f) return 0.0f;

    return inter / uni;
}

static std::vector<BBox> nms(const std::vector<BBox>& boxes, float iouThresh)
{
    std::vector<BBox> result;
    if (boxes.empty()) return result;

    // sorting by socre
    std::vector<BBox> sorted = boxes;
    std::sort(sorted.begin(), sorted.end(),
              [](const BBox& a, const BBox& b){ return a.score > b.score; });

    std::vector<bool> removed(sorted.size(), false);

    for (size_t i = 0; i < sorted.size(); ++i)
    {
        if (removed[i]) continue;
        const BBox& best = sorted[i];
        result.push_back(best);

        for (size_t j = i + 1; j < sorted.size(); ++j)
        {
            if (removed[j]) continue;
            if (sorted[j].cls_id != best.cls_id) continue; // different class will not suppress each other

            float iou = bboxIoU(best, sorted[j]);
            if (iou >= iouThresh)
            {
                removed[j] = true;
            }
        }
    }
    return result;
}

// ================== TensorRT encapsulate ==================

class TrtYolo
{
public:
    TrtYolo(const std::string& enginePath)
    : nbBindings_(0),
      inputIndex_(-1),
      mainOutIndex_(-1),
      inputSize_(0),
      mainOutSize_(0),
      stream_(0)
    {
        // import engine
        std::ifstream file(enginePath, std::ios::binary);
        if (!file)
        {
            std::cerr << "[TRT] can't import engine file：" << enginePath << std::endl;
            return;
        }
        file.seekg(0, file.end);
        size_t fsize = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engineData(fsize);
        file.read(engineData.data(), fsize);
        file.close();

        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_)
        {
            std::cerr << "[TRT] createInferRuntime failed" << std::endl;
            return;
        }

        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), fsize));
        if (!engine_)
        {
            std::cerr << "[TRT] deserializeCudaEngine failed" << std::endl;
            return;
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_)
        {
            std::cerr << "[TRT] createExecutionContext failed" << std::endl;
            return;
        }

        nbBindings_ = engine_->getNbBindings();
        std::cout << "[TRT] nbBindings = " << nbBindings_ << std::endl;

        deviceBuffers_.resize(nbBindings_, nullptr);
        elemCount_.resize(nbBindings_, 0);

        // scan all binding
        for (int i = 0; i < nbBindings_; ++i)
        {
            nvinfer1::Dims dims = engine_->getBindingDimensions(i);
            // if dynamic batch => batch=1
            if (dims.d[0] == -1)
            {
                dims.d[0] = 1;
                context_->setBindingDimensions(i, dims);
            }

            int64_t volume = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                volume *= dims.d[d];
            elemCount_[i] = volume;

            // GPU buffer
            cudaError_t err = cudaMalloc(&deviceBuffers_[i], volume * sizeof(float));
            if (err != cudaSuccess)
            {
                std::cerr << "[TRT] cudaMalloc failed，binding " << i
                          << "，error：" << cudaGetErrorString(err) << std::endl;
                return;
            }

            if (engine_->bindingIsInput(i))
            {
                inputIndex_ = i;
                inputSize_  = volume;

                std::cout << "[TRT] Input[" << i << "] dims: ";
                for (int d = 0; d < dims.nbDims; ++d) std::cout << dims.d[d] << " ";
                std::cout << std::endl;
            }
            else
            {
                std::cout << "[TRT] Output[" << i << "] dims: ";
                for (int d = 0; d < dims.nbDims; ++d) std::cout << dims.d[d] << " ";
                std::cout << std::endl;

                int lastDim = dims.d[dims.nbDims - 1];
                int targetLast = 5 + NUM_CLASSES;
                // main output
                if (lastDim == targetLast)
                {
                    mainOutIndex_ = i;
                    mainOutSize_  = volume;
                    std::cout << "[TRT] -> use Output[" << i
                              << "] as Nx(5+NUM_CLASSES) " << std::endl;
                }
            }
        }

        if (inputIndex_ < 0)
        {
            std::cerr << "[TRT] couldn't find input binding" << std::endl;
            return;
        }
        if (mainOutIndex_ < 0)
        {
            std::cerr << "[TRT] couldn't (1,N,5+NUM_CLASSES) output" << std::endl;
            return;
        }

        cudaStreamCreate(&stream_);
        std::cout << "[TRT] finished initialization" << std::endl;
    }

    ~TrtYolo()
    {
        for (int i = 0; i < nbBindings_; ++i)
        {
            if (deviceBuffers_[i])
                cudaFree(deviceBuffers_[i]);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool isOk() const
    {
        return (engine_ && context_ && inputIndex_ >= 0 && mainOutIndex_ >= 0);
    }

    bool infer(const cv::Mat& frame, std::vector<BBox>& outBoxes, cv::Mat& outVis, float& inferTimeSec)
    {
        if (!isOk()) return false;

        // pre-processing：resize + BGR->RGB + NCHW + /255
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(INPUT_W, INPUT_H));

        cv::Mat rgb;
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

        std::vector<float> inputData(inputSize_);
        int idx = 0;
        for (int c = 0; c < INPUT_C; ++c)
        {
            for (int h = 0; h < INPUT_H; ++h)
            {
                for (int w = 0; w < INPUT_W; ++w)
                {
                    cv::Vec3b pix = rgb.at<cv::Vec3b>(h, w);
                    float val = 0.0f;
                    if (c == 0)      val = pix[0];
                    else if (c == 1) val = pix[1];
                    else             val = pix[2];
                    inputData[idx++] = val / 255.0f;
                }
            }
        }

        // Host -> Device (input)
        cudaMemcpyAsync(deviceBuffers_[inputIndex_],
                        inputData.data(),
                        inputSize_ * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_);

        // prepare all binding index
        std::vector<void*> bindings(nbBindings_);
        for (int i = 0; i < nbBindings_; ++i)
        {
            bindings[i] = deviceBuffers_[i];  // non-null index
        }

        auto t1 = ros::Time::now();

        if (!context_->enqueueV2(bindings.data(), stream_, nullptr))
        {
            std::cerr << "[TRT] enqueueV2 failed" << std::endl;
            return false;
        }

        // 拷回主輸出
        std::vector<float> outputData(mainOutSize_);
        cudaMemcpyAsync(outputData.data(),
                        deviceBuffers_[mainOutIndex_],
                        mainOutSize_ * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream_);

        cudaStreamSynchronize(stream_);

        auto t2 = ros::Time::now();
        inferTimeSec = (t2 - t1).toSec();

        // 後處理：假設主輸出 shape = (1, N, 5+NUM_CLASSES)
        int elemPerBox = 5 + NUM_CLASSES;
        if (mainOutSize_ % elemPerBox != 0)
        {
            std::cerr << "[TRT] not divisible, elemPerBox，size=" << mainOutSize_
                      << ", elemPerBox=" << elemPerBox << std::endl;
            return false;
        }
        int numBoxes = mainOutSize_ / elemPerBox;

        int origH = frame.rows;
        int origW = frame.cols;
        float gainW = static_cast<float>(origW) / INPUT_W;
        float gainH = static_cast<float>(origH) / INPUT_H;

        std::vector<BBox> candBoxes;

        for (int i = 0; i < numBoxes; ++i)
        {
            const float* ptr = &outputData[i * elemPerBox];

            float cx = ptr[0];
            float cy = ptr[1];
            float w  = ptr[2];
            float h  = ptr[3];
            float objConf = ptr[4];

            float maxClsConf = -1.0f;
            int   maxClsId   = -1;
            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                float clsConf = ptr[5 + c];
                if (clsConf > maxClsConf)
                {
                    maxClsConf = clsConf;
                    maxClsId = c;
                }
            }

            float score = objConf * maxClsConf;
            if (score < CONF_THRESH) continue;

            float x1 = cx - w / 2.0f;
            float y1 = cy - h / 2.0f;
            float x2 = cx + w / 2.0f;
            float y2 = cy + h / 2.0f;

            x1 *= gainW;
            y1 *= gainH;
            x2 *= gainW;
            y2 *= gainH;

            BBox box;
            box.x1 = x1;
            box.y1 = y1;
            box.x2 = x2;
            box.y2 = y2;
            box.score = score;
            box.cls_id = maxClsId;
            candBoxes.push_back(box);
        }

        outBoxes = nms(candBoxes, IOU_THRESH);

        // draw bbox
        outVis = frame.clone();
        for (const auto& b : outBoxes)
        {
            int x1 = static_cast<int>(b.x1);
            int y1 = static_cast<int>(b.y1);
            int x2 = static_cast<int>(b.x2);
            int y2 = static_cast<int>(b.y2);

            std::string label = "id_" + std::to_string(b.cls_id);
            if (b.cls_id >= 0 && b.cls_id < (int)CLASS_NAMES.size())
                label = CLASS_NAMES[b.cls_id];

            cv::rectangle(outVis, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 2);

            char text[256];
            std::snprintf(text, sizeof(text), "%s %.2f", label.c_str(), b.score);
            cv::putText(outVis, text, cv::Point(x1, std::max(0, y1 - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        }

        return true;
    }

private:
    TrtLogger logger_;

    struct InferDeleter
    {
        template<typename T>
        void operator()(T* obj) const
        {
            if (obj) obj->destroy();
        }
    };

    std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context_;

    int nbBindings_;
    int inputIndex_;
    int mainOutIndex_;

    int64_t inputSize_;
    int64_t mainOutSize_;

    std::vector<void*>  deviceBuffers_;
    std::vector<int64_t> elemCount_;

    cudaStream_t stream_;
};

// ================== ROS Node：subscribe /usb_cam/image_raw rostopic ==================

class YoloRosNode
{
public:
    YoloRosNode(ros::NodeHandle& nh)
        : nh_(nh),
          it_(nh_),
          yolo_(ENGINE_PATH),
          frameIdx_(0),
          lastInferTime_(0.0)
    {
        if (!yolo_.isOk())
        {
            ROS_ERROR("TRT initialization failed，check engine version is compatible");
            ok_ = false;
            return;
        }
        
        ros::NodeHandle pnh("~");
        pnh.param("show_image", show_image_, false);  
        ROS_INFO("YOLO show_image = %s", show_image_ ? "true" : "false");

        // subscribe usb_cam image
        sub_ = it_.subscribe("/usb_cam/image_raw", 1,
                             &YoloRosNode::imageCallback, this);

        // publish YOLO detection result
        detPub_ = nh_.advertise<yolo_detection::detection>("/yolo/detection_node", 10);

        lastDispTime_ = ros::Time::now();
        
        if (show_image_)
            cv::namedWindow("YOLOv4 TRT Detection", cv::WINDOW_AUTOSIZE);

        ok_ = true;
        ROS_INFO("YOLOv4 TensorRT C++ node (subscribe /usb_cam/image_raw) has activated");
    }

    ~YoloRosNode()
    {
        if (show_image_)
            cv::destroyAllWindows();
    }

    bool isOk() const { return ok_; }

private:
    void imageCallback(const sensor_msgs::ImageConstPtr& msg)
    {
        // 1. ROS -> cv::Mat (BGR8)
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

        // 2. compute FPS
        ros::Time now = ros::Time::now();
        double dt = (now - lastDispTime_).toSec();
        lastDispTime_ = now;
        double dispFps = (dt > 1e-6) ? (1.0 / dt) : 0.0;

        cv::Mat showImg = frame.clone();

        // 3. inference every N consecutive frames
        if (frameIdx_ % DETECT_EVERY_N_FRAMES == 0)
        {
            lastBoxes_.clear();
            if (yolo_.infer(frame, lastBoxes_, showImg, lastInferTime_))
            {
                // inference succeed → publish detection
                for (const auto& b : lastBoxes_)
                {
                    yolo_detection::detection msgOut;

                    if (b.cls_id >= 0 && b.cls_id < (int)CLASS_NAMES.size())
                        msgOut.class_name = CLASS_NAMES[b.cls_id];
                    else
                        msgOut.class_name = "unknown";

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
            // draw last detection
            for (const auto& b : lastBoxes_)
            {
                int x1 = static_cast<int>(b.x1);
                int y1 = static_cast<int>(b.y1);
                int x2 = static_cast<int>(b.x2);
                int y2 = static_cast<int>(b.y2);

                std::string label = "id_" + std::to_string(b.cls_id);
                if (b.cls_id >= 0 && b.cls_id < (int)CLASS_NAMES.size())
                    label = CLASS_NAMES[b.cls_id];

                cv::rectangle(showImg, cv::Point(x1, y1), cv::Point(x2, y2),
                              cv::Scalar(0, 255, 0), 2);

                char text[256];
                std::snprintf(text, sizeof(text), "%s", label.c_str());
                cv::putText(showImg, text, cv::Point(x1, std::max(0, y1 - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6,
                            cv::Scalar(0, 255, 0), 2);
            }
        }

        // 4. show FPS on screen
        double inferFps = (lastInferTime_ > 1e-6) ? (1.0 / lastInferTime_) : 0.0;
        char fpsText[256];
        std::snprintf(fpsText, sizeof(fpsText), "Disp FPS: %.2f", dispFps);
        cv::putText(showImg, fpsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);
        std::snprintf(fpsText, sizeof(fpsText),
                      "Infer FPS: %.2f (every %d frames)", inferFps, DETECT_EVERY_N_FRAMES);
        cv::putText(showImg, fpsText, cv::Point(10, 60),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);

        // 5. show image
        if (show_image_)
        {
            cv::Mat disp;
            cv::resize(showImg, disp, cv::Size(320, 320));
            cv::imshow("YOLOv4 TRT Detection", disp);
            int key = cv::waitKey(1);
            if ((key & 0xFF) == 'q')
            {
                // press q to exit
                cv::destroyWindow("YOLOv4 TRT Detection");
             }
        }

    }

private:
    ros::NodeHandle& nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;
    ros::Publisher detPub_;

    TrtYolo yolo_;
    bool ok_ = false;
    bool show_image_;    

    int frameIdx_;
    std::vector<BBox> lastBoxes_;
    float lastInferTime_;
    ros::Time lastDispTime_;
};


// ================== main ==================

int main(int argc, char** argv)
{
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



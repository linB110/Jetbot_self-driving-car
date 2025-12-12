#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "utils.h"

struct YoloConfig
{
    std::string enginePath;
    int inputW;
    int inputH;
    int inputC;
    float confThresh;
    float iouThresh;
    std::vector<std::string> classNames;
};

class TrtYolo
{
public:
    explicit TrtYolo(const YoloConfig& config);
    ~TrtYolo();

    bool isOk() const;

    /**
     * @brief inference to single image
     * @param frame       iuput BGR image
     * @param outBoxes    outupt bbox pixel coordinate
     * @param outVis      draw bbox 
     * @param inferTimeSec inference (sec)
     */
    bool infer(const cv::Mat& frame,
               std::vector<BBox>& outBoxes,
               cv::Mat& outVis,
               float& inferTimeSec);

    const std::vector<std::string>& classNames() const { return classNames_; }
    int numClasses() const { return numClasses_; }

private:
    class TrtLogger : public nvinfer1::ILogger
    {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    struct InferDeleter
    {
        template<typename T>
        void operator()(T* obj) const
        {
            if (obj) obj->destroy();
        }
    };

private:
    bool initEngine(const std::string& enginePath);
    void releaseBuffers();

private:
    // config
    int inputW_;
    int inputH_;
    int inputC_;
    float confThresh_;
    float iouThresh_;
    std::vector<std::string> classNames_;
    int numClasses_;

    // TRT
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime,         InferDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine,      InferDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter> context_;

    int nbBindings_;
    int inputIndex_;
    int mainOutIndex_;

    int64_t inputSize_;
    int64_t mainOutSize_;

    std::vector<void*>   deviceBuffers_;
    std::vector<int64_t> elemCount_;

    cudaStream_t stream_;
};

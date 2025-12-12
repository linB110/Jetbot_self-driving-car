#include "trt_yolo.h"

#include <fstream>
#include <iostream>
#include <chrono>

void TrtYolo::TrtLogger::log(Severity severity, const char* msg) noexcept
{
    if (severity <= Severity::kWARNING)
    {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

TrtYolo::TrtYolo(const YoloConfig& config)
    : inputW_(config.inputW),
      inputH_(config.inputH),
      inputC_(config.inputC),
      confThresh_(config.confThresh),
      iouThresh_(config.iouThresh),
      classNames_(config.classNames),
      numClasses_(static_cast<int>(config.classNames.size())),
      nbBindings_(0),
      inputIndex_(-1),
      mainOutIndex_(-1),
      inputSize_(0),
      mainOutSize_(0),
      stream_(0)
{
    if (numClasses_ <= 0)
    {
        std::cerr << "[TRT] numClasses=0，please check num classes is correct" << std::endl;
        return;
    }

    if (!initEngine(config.enginePath))
    {
        std::cerr << "[TRT] TensorRT engine initialization failed" << std::endl;
        return;
    }

    // warmup
    cv::Mat dummy(inputH_, inputW_, CV_8UC3, cv::Scalar(0,0,0));
    std::vector<BBox> dummyBoxes;
    cv::Mat dummyVis;
    float t;
    std::cout << "[TRT] warmup inferencing..." << std::endl;
    infer(dummy, dummyBoxes, dummyVis, t);
    std::cout << "[TRT] warmup done, first inference time : " << t << " s" << std::endl;
}

TrtYolo::~TrtYolo()
{
    releaseBuffers();
}

bool TrtYolo::initEngine(const std::string& enginePath)
{
    // read engine
    std::ifstream file(enginePath, std::ios::binary);
    if (!file)
    {
        std::cerr << "[TRT] can't open engine file: " << enginePath << std::endl;
        return false;
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
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), fsize));
    if (!engine_)
    {
        std::cerr << "[TRT] deserializeCudaEngine failed" << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_)
    {
        std::cerr << "[TRT] createExecutionContext failed" << std::endl;
        return false;
    }

    nbBindings_ = engine_->getNbBindings();
    std::cout << "[TRT] nbBindings = " << nbBindings_ << std::endl;

    deviceBuffers_.resize(nbBindings_, nullptr);
    elemCount_.resize(nbBindings_, 0);

    // alloc GPU buffer
    for (int i = 0; i < nbBindings_; ++i)
    {
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);

        // dynamic shape (batch) -> set to 1
        if (dims.d[0] == -1)
        {
            dims.d[0] = 1;
            context_->setBindingDimensions(i, dims);
        }

        int64_t volume = 1;
        for (int d = 0; d < dims.nbDims; ++d)
        {
            volume *= dims.d[d];
        }

        elemCount_[i] = volume;

        cudaError_t err = cudaMalloc(&deviceBuffers_[i], volume * sizeof(float));
        if (err != cudaSuccess)
        {
            std::cerr << "[TRT] cudaMalloc failed，binding " << i
                      << "，error: " << cudaGetErrorString(err) << std::endl;
            return false;
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
            int targetLast = 5 + numClasses_;

            if (lastDim == targetLast)
            {
                mainOutIndex_ = i;
                mainOutSize_  = volume;
                std::cout << "[TRT] -> use Output[" << i
                          << "] as Nx(5+NUM_CLASSES)" << std::endl;
            }
        }
    }

    if (inputIndex_ < 0)
    {
        std::cerr << "[TRT] couldn't find input binding" << std::endl;
        return false;
    }
    if (mainOutIndex_ < 0)
    {
        std::cerr << "[TRT] couldn't find (1,N,5+NUM_CLASSES) output" << std::endl;
        return false;
    }

    cudaStreamCreate(&stream_);
    std::cout << "[TRT] finished initialization" << std::endl;

    return true;
}

void TrtYolo::releaseBuffers()
{
    for (int i = 0; i < nbBindings_; ++i)
    {
        if (deviceBuffers_[i])
        {
            cudaFree(deviceBuffers_[i]);
            deviceBuffers_[i] = nullptr;
        }
    }
    if (stream_)
    {
        cudaStreamDestroy(stream_);
        stream_ = 0;
    }
}

bool TrtYolo::isOk() const
{
    return (engine_ && context_ && inputIndex_ >= 0 && mainOutIndex_ >= 0);
}

bool TrtYolo::infer(const cv::Mat& frame,
                    std::vector<BBox>& outBoxes,
                    cv::Mat& outVis,
                    float& inferTimeSec)
{
    if (!isOk()) return false;

    // 1. preprocess: resize + BGR->RGB + NCHW + normalize [0,1]
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputW_, inputH_));

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    std::vector<float> inputData(inputSize_);
    int idx = 0;
    for (int c = 0; c < inputC_; ++c)
    {
        for (int h = 0; h < inputH_; ++h)
        {
            for (int w = 0; w < inputW_; ++w)
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

    // 2. copy H2D
    cudaMemcpyAsync(deviceBuffers_[inputIndex_],
                    inputData.data(),
                    inputSize_ * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream_);

    std::vector<void*> bindings(nbBindings_);
    for (int i = 0; i < nbBindings_; ++i)
    {
        bindings[i] = deviceBuffers_[i];
    }

    auto t1 = std::chrono::steady_clock::now();

    if (!context_->enqueueV2(bindings.data(), stream_, nullptr))
    {
        std::cerr << "[TRT] enqueueV2 failed" << std::endl;
        return false;
    }

    // 3. copy D2H
    std::vector<float> outputData(mainOutSize_);
    cudaMemcpyAsync(outputData.data(),
                    deviceBuffers_[mainOutIndex_],
                    mainOutSize_ * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_);

    cudaStreamSynchronize(stream_);

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<float> dur = t2 - t1;
    inferTimeSec = dur.count();

    int elemPerBox = 5 + numClasses_;
    if (mainOutSize_ % elemPerBox != 0)
    {
        std::cerr << "[TRT] mainOutSize not divisible by elemPerBox, size="
                  << mainOutSize_ << ", elemPerBox=" << elemPerBox << std::endl;
        return false;
    }

    int numBoxes = mainOutSize_ / elemPerBox;

    int origH = frame.rows;
    int origW = frame.cols;
    float gainW = static_cast<float>(origW) / static_cast<float>(inputW_);
    float gainH = static_cast<float>(origH) / static_cast<float>(inputH_);

    std::vector<BBox> candBoxes;
    candBoxes.reserve(numBoxes);

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
        for (int c = 0; c < numClasses_; ++c)
        {
            float clsConf = ptr[5 + c];
            if (clsConf > maxClsConf)
            {
                maxClsConf = clsConf;
                maxClsId = c;
            }
        }

        float score = objConf * maxClsConf;
        if (score < confThresh_) continue;

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

    outBoxes = nms(candBoxes, iouThresh_);

    outVis = frame.clone();
    for (const auto& b : outBoxes)
    {
        int x1 = static_cast<int>(b.x1);
        int y1 = static_cast<int>(b.y1);
        int x2 = static_cast<int>(b.x2);
        int y2 = static_cast<int>(b.y2);

        std::string label = "id_" + std::to_string(b.cls_id);
        if (b.cls_id >= 0 && b.cls_id < static_cast<int>(classNames_.size()))
            label = classNames_[b.cls_id];

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

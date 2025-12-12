#pragma once

#include <vector>

struct BBox
{
    float x1, y1, x2, y2;
    float score;
    int   cls_id;
};

float bboxIoU(const BBox& a, const BBox& b);
std::vector<BBox> nms(const std::vector<BBox>& boxes, float iouThresh);

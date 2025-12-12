#include "utils.h"
#include <algorithm>
#include <cmath>

float bboxIoU(const BBox& a, const BBox& b)
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

std::vector<BBox> nms(const std::vector<BBox>& boxes, float iouThresh)
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
            if (sorted[j].cls_id != best.cls_id) continue;// different class will not suppress each other

            float iou = bboxIoU(best, sorted[j]);
            if (iou >= iouThresh)
            {
                removed[j] = true;
            }
        }
    }
    return result;
}

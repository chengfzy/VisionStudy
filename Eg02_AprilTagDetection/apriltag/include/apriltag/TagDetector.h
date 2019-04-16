#ifndef TAGDETECTOR_H
#define TAGDETECTOR_H

#include <vector>

#include "opencv2/opencv.hpp"

#include "apriltag/FloatImage.h"
#include "apriltag/TagDetection.h"
#include "apriltag/TagFamily.h"

namespace apriltag {

class TagDetector {
   public:
    const TagFamily thisTagFamily;

    //! Constructor
    // note: TagFamily is instantiated here from TagCodes
    TagDetector(const TagCodes& tagCodes, const size_t blackBorder = 2) : thisTagFamily(tagCodes, blackBorder) {}

    std::vector<TagDetection> extractTags(const cv::Mat& image);
};

}  // namespace apriltag

#endif

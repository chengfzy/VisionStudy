#pragma once
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

// get distance between 2 points
template <typename Type>
double pointDist(const cv::Point_<Type>& p1, const cv::Point_<Type> p2) {
    cv::Point_<Type> dp = p1 - p2;
    return std::sqrt(dp.x * dp.x + dp.y * dp.y);
}

struct QuadCountour {
    std::array<cv::Point, 4> pts;
    int parentContour;

    QuadCountour(const std::array<cv::Point, 4> points, int parentContour_) : parentContour(parentContour_) {
        pts = points;
    }
};

// Chessboard Corner
struct ChessBoardCorner {
    cv::Point2f pt;                  // coordinates of the corner
    int row;                         // board row index
    int count;                       // number of neighbor corners
    ChessBoardCorner* neighbors[4];  // neighbors corners

    ChessBoardCorner(const cv::Point2f& pt_ = cv::Point2f()) : pt(pt_), row(0), count(0) {
        neighbors[0] = neighbors[1] = neighbors[2] = neighbors[3] = nullptr;
    }

    float sumDist(int& n) const {
        float sum{0};
        n = 0;
        for (int i = 0; i < 4; ++i) {
            if (neighbors[i]) {
                sum += pointDist(neighbors[i]->pt, pt);
                ++n;
            }
        }
        return sum;
    }
};

// Chessboard Quadrangle
struct ChessBoardQuad {
    int count;      // number of quad neighbors
    int groundIdx;  // quad ground ID
    int row;        // row of this quad
    int col;        // col of this quad
    bool ordered;   // true if corners/neighbors are ordered counter-clockwise
    float edgeLen;  // quad edge len, in pix^2

    ChessBoardCorner* corners[4];  // coordinates of quad corners
    ChessBoardQuad* neighbors[4];  // pointers of quad neighbors

    ChessBoardQuad(int groundIdx_ = -1) : count(0), groundIdx(groundIdx_), row(0), col(0), ordered(false), edgeLen(0) {
        corners[0] = corners[1] = corners[2] = corners[3] = nullptr;
        neighbors[0] = neighbors[1] = neighbors[2] = neighbors[3] = nullptr;
    }
};

// chessboard detector
class ChessBoardDetector {
   public:
    ChessBoardDetector(const cv::Size& patternSize) : patternSize_(patternSize), quadsCount_(0) {}

   public:
    void generateQuads(const cv::Mat& image, bool filterQuads = true);

    // bool processQuads(std::vector<cv::Point2f>& outCorners, int& prevSize);

   private:
    cv::Mat binaryImage_;
    cv::Size patternSize_;
    std::vector<ChessBoardQuad> quads_;
    std::vector<ChessBoardCorner> corners_;
    std::size_t quadsCount_;
};

// generate quads
void ChessBoardDetector::generateQuads(const cv::Mat& image, bool filterQuads) {
    binaryImage_ = image;
    quads_.clear();
    corners_.clear();

    int minSize{25};  // empiric bound for minimal allowed size(ares) for squares

    // find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    // check size
    if (contours.empty()) {
        LOG(INFO) << "cannot find any contours";
        return;
    }

    std::vector<int> contourChildCounter(contours.size(), 0);
    std::vector<QuadCountour> contourQuads;
    int borderIdx{-1};

    for (size_t n = contours.size() - 1; n >= 0; --n) {
        auto& parentIdx = hierarchy[n][3];
        // holes only (no child contours, have parent)
        if (hierarchy[n][2] != -1 || parentIdx == -1) {
            continue;
        }

        // check area(size)
        const auto& contour = contours[n];
        cv::Rect contourRect = cv::boundingRect(contour);
        if (contourRect.area() < minSize) {
            continue;
        }

        // approximate a polygonal curves to filter this contours
        std::vector<cv::Point> approxContour;
        for (int approxLevel = 1; approxLevel <= 7; ++approxLevel) {
            cv::approxPolyDP(contour, approxContour, approxLevel, true);
            if (approxContour.size() == 4) {
                break;
            }

            // we call this again on its own output, because sometimes approxPoly() does not simplify as much as it
            // should
            std::vector<cv::Point> approxContourTmp;
            std::swap(approxContour, approxContourTmp);
            cv::approxPolyDP(approxContourTmp, approxContour, approxLevel, true);
            if (approxContour.size() == 4) {
                break;
            }
        }

        // reject non-quadrengles
        if (approxContour.size() != 4) {
            continue;
        }
        if (!cv::isContourConvex(approxContour)) {
            continue;
        }

        // convert the contour points to array
        std::array<cv::Point, 4> pts;
        for (size_t i = 0; i < 4; ++i) {
            pts[i] = approxContour[i];
        }

        // filter this candidates
        if (filterQuads) {
            double p = cv::arcLength(pts, true);
            double area = cv::contourArea(pts, true);

            double d1 = pointDist(pts[0], pts[2]);
            double d2 = pointDist(pts[1], pts[3]);
            double d3 = pointDist(pts[0], pts[1]);
            double d4 = pointDist(pts[1], pts[2]);
            // only accept those quadrangles which are more square that rectanglar and which are big enough
            if (!(d3 * 4 > d4 && d4 * 4 > d3 && d3 * d4 < 0.15 * area && area > minSize && d1 >= 0.15 * p &&
                  d2 >= 0.15 * p)) {
                continue;
            }
        }

        ++contourChildCounter[parentIdx];
        if (borderIdx != parentIdx &&
            (borderIdx < 0 || contourChildCounter[borderIdx] < contourChildCounter[parentIdx])) {
            borderIdx = parentIdx;
        }
        contourQuads.emplace_back(QuadCountour(pts, parentIdx));
    }

    size_t quadNum = contourQuads.size();
    size_t maxQuadBufSize = std::max(static_cast<size_t>(2), quadNum * 3);
    quads_.resize(maxQuadBufSize);
    corners_.resize(maxQuadBufSize * 4);

    // create array of quads structures
    size_t quadCount{0};
    for (size_t n = 0; n < quadNum; ++n) {
        QuadCountour& qc = contourQuads[n];
        if (filterQuads && qc.parentContour != borderIdx) {
            continue;
        }

        // reset group ID
        size_t quadIdx = quadCount;
        ++quadCount;
        ChessBoardQuad& q = quads_[quadIdx];
        q = ChessBoardQuad();

        for (size_t i = 0; i < 4; ++i) {
            corners_[4 * quadIdx + i] = ChessBoardCorner(qc.pts[i]);
            q.corners[i] = &corners_[4 * quadIdx + i];
        }
        // set edge length
        q.edgeLen = 999999;  // initialize to some big value
        for (size_t i = 0; i < 4; ++i) {
            float d = static_cast<float>(pointDist(q.corners[i]->pt, q.corners[(i + 1) & 3]->pt));
            q.edgeLen = std::min(q.edgeLen, d);
        }
    }

    quadsCount_ = quadCount;
}
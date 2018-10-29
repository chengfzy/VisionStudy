#include <iostream>
#include "ChessBoardDetector.hpp"
#include "common/common.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace common;

DEFINE_string(calibFile, "../../data/calib01.jpg", "calibration file");

// find corners using OpenCV method
void openCVMethod(const Mat& image) {
    // find and draw the chessboard
    vector<Point2f> corners;
    Size patternSize(9, 6);
    bool found = findChessboardCorners(image, patternSize, corners,
                                       CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK);
    drawChessboardCorners(image, patternSize, corners, found);
    imshow("OpenCV Chessboard", image);
    waitKey();
}

// perform smart image thresholding based on analysis of intensity histogram
void histogramBasedBinary(Mat& image) {
    const int kCols = image.cols;
    const int kRows = image.rows;
    const int kMaxPix = kCols * kRows;
    const int kMaxPix1 = kMaxPix / 100;
    const int kNumBins{256};
    const int kMaxPos{20};

    array<int, kNumBins> histIntensity{0};
    array<int, kNumBins> histSmooth{0};
    array<int, kNumBins> histGrad{0};
    array<int, kMaxPos> maxPos{0};

    // compute intensity histogram
    histIntensity.fill(0);
    for (int r = 0; r < image.rows; ++r) {
        const uchar* row = image.ptr<uchar>(r);
        for (int c = 0; c < image.cols; ++c) {
            ++histIntensity[row[c]];
        }
    }

    // smooth the histogram using window of size 2 * kWidth + 1
    const int kWidth{1};
    for (int i = 0; i < kNumBins; ++i) {
        int idxMin = std::max(0, i - kWidth);
        int idxMax = std::min(kNumBins - 1, i + kWidth);
        int smooth{0};
        for (int idx = idxMin; idx <= idxMax; ++idx) {
            smooth += histIntensity[idx];
        }
        histSmooth[i] = smooth / (2 * kWidth + 1);
    }

    // compute histogram gradient
    histGrad[0] = 0;
    int preGrad{0};
    for (int i = 1; i < kNumBins - 1; ++i) {
        int grad = histSmooth[i - 1] - histSmooth[i + 1];
        if (std::abs(grad) < 100) {
            if (preGrad == 0) {
                grad = -100;
            } else {
                grad = preGrad;
            }
        }
        histGrad[i] = grad;
        preGrad = grad;
    }
    histGrad[kNumBins - 1] = 0;

    // print for debug
    cout << section("Histogram Info") << endl;
    for (size_t i = 0; i < kNumBins; ++i) {
        cout << "[" << i << "] " << histIntensity[i] << ", " << histSmooth[i] << ", " << histGrad[i] << endl;
    }

    // check for zeros
    unsigned maximaCount{0};
    for (int i = kNumBins - 2; i > 2 && maximaCount < kMaxPos; --i) {
        if (histGrad[i - 1] < 0 && histGrad[i] > 0) {
            int sumArroundMax = histSmooth[i - 1] + histSmooth[i] + histSmooth[i + 1];
            if (sumArroundMax >= kMaxPix1 || i >= 64) {
                maxPos[maximaCount++] = i;
            }
        }
    }

    int thresh{0};
    if (maximaCount == 0) {
        // not any maxima inside (only 0 and 255 which are not counted above)
        // Does image black-write already
        const int maxPix2 = kMaxPix1 / 2;
        // select mean intensity
        for (int sum = 0, i = 0; i < kNumBins; ++i) {
            sum += histIntensity[i];
            if (sum > maxPix2) {
                thresh = i;
                break;
            }
        }
    } else if (maximaCount == 1) {
        thresh = maxPos[0] / 2;
    } else if (maximaCount == 2) {
        thresh = (maxPos[0] + maxPos[1]) / 2;
    } else {  // maximaCount >= 3
        // check threshold for white
        int idxAccSum{0};
        int accum{0};
        for (int i = kNumBins - 1; i > 0; --i) {
            accum += histIntensity[i];
            // 1/18 ~ 5.5%, the minimum required number of pixels required for white part of chessboard
            if (accum > kMaxPix / 18) {
                idxAccSum = i;
                break;
            }
        }

        unsigned idxBGMax{0};
        int brightMax = maxPos[0];
        for (unsigned i = 0; i < maximaCount - 1; ++i) {
            idxBGMax = i + 1;
            if (maxPos[i] < idxAccSum) {
                break;
            }
            brightMax = maxPos[i];
        }

        // check threshold for black
        int maxVal = histIntensity[maxPos[idxBGMax]];

        // if too close too 255, jump to next maximum
        if (maxPos[idxBGMax] >= 250 && idxBGMax + 1 < maximaCount) {
            ++idxBGMax;
            maxVal = histIntensity[maxPos[idxBGMax]];
        }

        for (unsigned i = idxBGMax + 1; i < maximaCount; ++i) {
            if (histIntensity[maxPos[i]] >= maxVal) {
                maxVal = histIntensity[maxPos[i]];
                idxBGMax = 0;
            }
        }

        // setting threshold for binarization
        int dist2 = (brightMax - maxPos[idxBGMax]) / 2;
        thresh = brightMax - dist2;
    }

    if (thresh > 0) {
        image = image >= thresh;
    }
}

void getQuadrangleHypotheses(const vector<vector<Point>>& contours, const vector<Vec4i>& hierarchy,
                             vector<pair<float, int>>& quads, int classId) {
    const float minAspectRatio{0.3f};
    const float maxAspectRatio{3.0f};
    const float minBoxSize{10.0f};

    for (size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy[i][3] != -1) {  // skip holes
            continue;
        }

        RotatedRect box = minAreaRect(contours[i]);

        float boxSize = max(box.size.width, box.size.height);
        if (boxSize < minBoxSize) {
            continue;
        }
        float aspectRatio = box.size.width / max(box.size.height, 1.f);
        if (aspectRatio < minAspectRatio || aspectRatio > maxAspectRatio) {
            continue;
        }

        quads.push_back(std::pair<float, int>(boxSize, classId));
    }
}

bool checkQuads(vector<pair<float, int>>& quads, const Size& size) {
    const size_t minQuadCount = size.width * size.height / 2;
    sort(quads.begin(), quads.end(),
         [](const pair<float, int>& p1, const pair<float, int>& p2) { return p1.first < p2.first; });

    // check if there are mnay hypotheses with similar sizes
    const float sizeRelDev{0.4f};

    for (size_t i = 0; i < quads.size(); ++i) {
        size_t j = i + 1;
        for (; j < quads.size(); ++j) {
            if (quads[j].first / quads[i].first > 1.0f + sizeRelDev) {
                break;
            }
        }

        if (j + 1 > minQuadCount + 1) {
            // check the number of black and white squares
            vector<int> counts{0, 0};
            for (size_t n = i; n != j; ++n) {
                ++counts[quads[n].second];
            }
            const int blackCount = cvRound(ceil(size.width / 2.0) * ceil(size.height / 2.0));
            const int whiteCount = cvRound(floor(size.width / 2.0) * floor(size.height / 2.0));
            if (counts[0] < blackCount * 0.75 || counts[1] < whiteCount * 0.75) {
                continue;
            }
            return true;
        }
    }
    return false;
}

void fillQuad(Mat& white, Mat& black, double whiteThresh, double blackThresh, vector<pair<float, int>>& quads) {
    Mat thresh;

    // for white image
    {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        threshold(white, thresh, whiteThresh, 255, THRESH_BINARY);
        findContours(thresh, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        getQuadrangleHypotheses(contours, hierarchy, quads, 1);
    }

    // for black image
    {
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        threshold(black, thresh, blackThresh, 255, THRESH_BINARY_INV);
        findContours(thresh, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
        getQuadrangleHypotheses(contours, hierarchy, quads, 0);
    }
}

// fast check if a chessboard is in the input image, use binary image
bool checkChessboardBinary(const Mat& image, const Size& size) {
    Mat white = image.clone();
    Mat black = image.clone();

    for (int erosionCount = 0; erosionCount <= 3; ++erosionCount) {
        if (0 != erosionCount) {  // first iteration keeps original images
            erode(white, white, Mat());
            dilate(black, black, Mat());
        }

        vector<pair<float, int>> quads;
        fillQuad(white, black, 128, 128, quads);
        if (checkQuads(quads, size)) {
            return true;
        }
    }

    return false;
}

// check if a chess board is in the input image, use raw image
bool checkChessboard(const Mat& image, const Size& size) {
    const float blackLevel{20.f};
    const float whiteLevel{130.f};
    const float blackWhiteGap{70.f};

    Mat white;
    Mat black;
    erode(image, white, Mat());
    dilate(image, black, Mat());

    for (float threshLevel = blackLevel; threshLevel < whiteLevel; threshLevel += 20.0f) {
        vector<pair<float, int>> quads;
        fillQuad(white, black, threshLevel + blackWhiteGap, threshLevel, quads);
        if (checkQuads(quads, size)) {
            return true;
        }
    }

    return false;
}

// find corners using my method
void findChessboard(Mat& image, bool fastCheck = true) {
    Size patternSize(9, 6);

    // convert to gray image
    if (image.channels() != 1) {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }

    // binarization
    Mat binaryImg = image.clone();
    histogramBasedBinary(binaryImg);
    imshow("Binary Image", binaryImg);

    // fast check
    if (fastCheck) {
        if (!checkChessboardBinary(binaryImg, patternSize)) {  // check fail, back to the old method
            if (!checkChessboard(image, patternSize)) {
                LOG(ERROR) << "cannot find chessboard in the input image";
                return;
            }
        }
    }

    bool found{false};

    //  try standard dilation, but if the pattern is not found, iterate the whole procedure with higher dilations. This
    //  is necessary because some squares simply do not separate properly with a single dilation. However, we want to
    //  use the minimum number of dilations possible since dilations cause the squares to become smaller, making it
    //  difficult to detect smaller squares
    ChessBoardDetector detector(patternSize);
    vector<Point2f> outCorners;
    int preSize{0};
    for (int i = 0; i < 7; ++i) {
        // use binary image
        dilate(binaryImg, binaryImg, Mat());

        imshow("Before Add Boarder", binaryImg);

        // so we can find rectangles that go to the edge, we draw a white line around the image edge. Otherwise
        // FindContours will miss those clipped rectangle contours. The border color will be the image mean, because
        // other wise we risk screwing up filters like Smooth()
        rectangle(binaryImg, Point(0, 0), Point(binaryImg.cols - 1, binaryImg.rows - 1), Scalar(255, 255, 255), 3,
                  LINE_8);
        imshow("Add White Line", binaryImg);

        //detector.generateQuads(binaryImg, true);
        break;
        //if (detector.processQuads(outCorners, preSize)) {
        //    found = true;
        //    break;
        //}
    }

    waitKey();
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    // open file
    Mat calibImg = imread(FLAGS_calibFile, IMREAD_UNCHANGED);
    if (calibImg.empty()) {
        LOG(FATAL) << "cannot open image \"" << FLAGS_calibFile << "\"";
        return -1;
    }

    // find and draw the chessboard use OpenCV method
    // openCVMethod(calibImg);
    findChessboard(calibImg);

    google::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}
#include <iostream>
#include "apriltag/Tag16h5.h"
#include "apriltag/Tag25h7.h"
#include "apriltag/Tag25h9.h"
#include "apriltag/Tag36h11.h"
#include "apriltag/Tag36h9.h"
#include "apriltag/TagDetector.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;
using namespace apriltag;

DEFINE_string(tagType, "36h11", "AprilTag type, and only could be \"16h5, 25h7, 25h9, 36h9, 36h11\"");
DEFINE_bool(fromImage, true, "true for detect from image file, false for detect from video");
DEFINE_string(imageFile, "../../data/AprilTag_01.jpeg", "image file with AprilTag");

// draw detection in image
void drawDetections(Mat& image, const vector<TagDetection>& detections) {
    // print out detection
    for (auto& v : detections) {
        cout << "ID = " << v.id << ", Hamming =  " << v.hammingDistance << ", Rotation = " << v.rotation
             << ", XY Orientation = " << v.getXYOrientation() / M_PI * 180 << " deg" << endl;
        v.draw(image);
    }

    // draw the original point
    if (!detections.empty()) {
        circle(image, Point2f(detections[0].p[0].first, detections[0].p[0].second), 3, Scalar(0, 0, 255), 2, LINE_AA);
    }
}

// detect AprilTag from image file
void detectFromImage(TagDetector& detector, const string& imageFile) {
    // read image from file
    Mat image = imread(imageFile, IMREAD_UNCHANGED);
    if (image.empty()) {
        LOG(FATAL) << "cannot open image file \"" << imageFile << "\"";
        return;
    }

    // resize image if it's too large
    if (image.cols > 1080) {
        double scale = 1080.f / image.cols;
        resize(image, image, Size(), scale, scale);
    }

    // convert image to gray
    Mat gray;
    if (image.type() == CV_8UC4) {
        cvtColor(image, gray, COLOR_BGRA2GRAY);
    } else if (image.type() == CV_8UC3) {
        cvtColor(image, gray, COLOR_BGR2GRAY);
    } else if (image.type() == CV_8UC1) {
        gray = image;
    } else {
        LOG(FATAL) << "unsupported image type";
        return;
    }

    // detect
    vector<TagDetection> detections = detector.extractTags(gray);
    cout << detections.size() << " tags detected" << endl;

    // draw detections
    drawDetections(image, detections);

    // show image
    imshow("AprilTag Image", image);
    waitKey();
}

// detect AprilTag from video
void detectFromVideo(TagDetector& detector) {
    // open camera
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        LOG(FATAL) << "cannot open camera";
        return;
    }

    while (true) {
        // read frame and convert to gray
        Mat frame;
        capture.read(frame);
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // detect
        vector<TagDetection> detections = detector.extractTags(gray);
        // draw detections
        drawDetections(frame, detections);

        // show image
        imshow("AprilTag Image", frame);

        // wait key to exit
        int pressKey = waitKey(30);
        if (27 == pressKey || 'q' == pressKey || 'Q' == pressKey || 'x' == pressKey || 'X' == pressKey ||
            ' ' == pressKey) {
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    // parse tag type
    TagCodes tagType{tagCodes36h11};
    if (FLAGS_tagType == "16h5") {
        tagType = tagCodes16h5;
    } else if (FLAGS_tagType == "25h7") {
        tagType = tagCodes25h7;
    } else if (FLAGS_tagType == "25h9") {
        tagType = tagCodes25h9;
    } else if (FLAGS_tagType == "36h9") {
        tagType = tagCodes36h9;
    } else if (FLAGS_tagType == "36h11") {
        tagType = tagCodes36h11;
    } else {
        LOG(FATAL) << "unrecognized tag type";
        return -1;
    }

    // construct detector
    TagDetector detector(tagType, 1);

    // detect and show
    if (FLAGS_fromImage) {
        detectFromImage(detector, FLAGS_imageFile);
    } else {
        detectFromVideo(detector);
    }

    google::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}
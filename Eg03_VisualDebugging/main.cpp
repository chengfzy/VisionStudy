#include <iostream>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

//#define CVVISUAL_DEBUGMODE
#include "opencv2/cvv.hpp"

using namespace std;
using namespace cv;

template <typename T>
std::string toString(const T& p) {
    std::stringstream ss;
    ss << p;
    return ss.str();
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true;

    // open camera
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        LOG(FATAL) << "cannot open camera";
        return -1;
    }

    Mat prevImgGray;
    vector<KeyPoint> prevKeypoints;
    Mat prevDescriptors;
    Ptr<ORB> detector = ORB::create(500);
    BFMatcher matcher(NORM_HAMMING);

    for (size_t n = 0; n < 10; ++n) {
        Mat frame;
        capture >> frame;
        cout << n << ": frame captured" << endl;

        // show frame
        string imgString = "Frame " + toString(n);
        cvv::showImage(frame, CVVISUAL_LOCATION, imgString, "");

        // convert to gray
        Mat imgGray;
        cvtColor(frame, imgGray, COLOR_BGR2GRAY);
        cvv::debugFilter(frame, imgGray, CVVISUAL_LOCATION, "To Gray");

        // detect ORB features
        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detectAndCompute(imgGray, noArray(), keypoints, descriptors);
        cout << n << ": detected " << keypoints.size() << " keypoints" << endl;

        // match them to previous image
        if (!prevImgGray.empty()) {
            vector<DMatch> matches;
            matcher.match(prevDescriptors, descriptors, matches);
            cout << n << ": all matches size = " << matches.size() << endl;
            string matchString = "All Matches " + toString(n - 1) + " <-> " + toString(n);
            cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION, matchString,
                             "");

            // remove worst
            sort(matches.begin(), matches.end());
            matches.resize(static_cast<size_t>(0.8 * matches.size()));
            cout << n << ": best matches size = " << matches.size();
            string bestMatchString = "Best 0.8 Matches " + toString(n - 1) + " <-> " + toString(n);
            cvv::debugDMatch(prevImgGray, prevKeypoints, imgGray, keypoints, matches, CVVISUAL_LOCATION,
                             bestMatchString, "");
        }

        prevImgGray = imgGray;
        prevKeypoints = keypoints;
        prevDescriptors = descriptors;
    }

    cvv::finalShow();

    google::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}
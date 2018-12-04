#include <iostream>
#include <opencv2/imgproc.hpp>
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "opencv2/highgui.hpp"
#include "zbar.h"

using namespace std;
using namespace cv;

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

    // init zbar
    zbar::ImageScanner scanner;
    scanner.set_config(zbar::ZBAR_QRCODE, zbar::ZBAR_CFG_ENABLE, 1);

    while (true) {
        Mat frame;
        capture.read(frame);

        // convert to gray
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // wrap image data
        zbar::Image image(static_cast<unsigned>(gray.cols), static_cast<unsigned>(gray.rows), "Y800", gray.data,
                          static_cast<unsigned long>(gray.cols * gray.rows));

        // detect image
        int codeCount = scanner.scan(image);
        for (auto it = image.symbol_begin(); it != image.symbol_end(); ++it) {
            cout << "Code Name: " << it->get_type_name() << ", Info: " << it->get_data() << endl;

            // draw symbols
            if (it->get_location_size() == 4) {
                for (unsigned i = 0; i < 4; ++i) {
                    line(frame, Point(it->get_location_x(i), it->get_location_y(i)),
                         Point(it->get_location_x((i + 1) % 4), it->get_location_y((i + 1) % 4)), Scalar(0, 255, 0), 2,
                         LINE_AA);
                }
                circle(frame, Point(it->get_location_x(0), it->get_location_y(0)), 5, Scalar(0, 0, 255), 2, LINE_AA);
            }
        }

        // show image
        imshow("QR Code", frame);

        // wait key to exit
        int pressKey = waitKey(30);
        if (27 == pressKey || 'q' == pressKey || 'Q' == pressKey || 'x' == pressKey || 'X' == pressKey ||
            ' ' == pressKey) {
            break;
        }
    }

    google::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();
    return 0;
}
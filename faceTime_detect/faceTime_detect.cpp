#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>

#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logtag.hpp>
#include <opencv2/core/utils/logger.hpp> 
#define OPENCV_LOGGER_DEFINES_HPP

using namespace cv;
using namespace std;

Mat img, tmpimg, handImg;

void detect_Ycrcb();

int main()
{
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

    img = imread("good.jpg");

	if (img.empty()) {
		cout << "이미지를 읽을 수 없습니다." << endl;
		exit(-1);
	}

	imshow("image", img);
	detect_Ycrcb();

	waitKey(0);

	return 0;
}

void detect_Ycrcb() {

    Mat ycrcbImg;

    cvtColor(img, ycrcbImg, COLOR_BGR2YCrCb); 
    inRange(ycrcbImg, Scalar(0, 133, 77), Scalar(255, 173, 127), ycrcbImg); 

    handImg = Mat::zeros(img.size(), CV_8UC3); 

    bitwise_and(img, img, handImg, ycrcbImg); 

    imshow("handImage", handImg); 

    if (handImg.empty()) {
        cout << "이미지를 읽을 수 없습니다." << endl;
        exit(-1);
    } 
}

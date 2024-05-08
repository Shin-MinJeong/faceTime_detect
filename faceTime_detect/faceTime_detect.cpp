#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/core.hpp>

#include <opencv2/core/utils/logger.defines.hpp>
#include <opencv2/core/utils/logtag.hpp>
#include <opencv2/core/utils/logger.hpp> 


#define OPENCV_LOGGER_DEFINES_HPP

using namespace cv;
using namespace std;

Mat img, tmpimg, skinImg, handImg;

void detect_Ycrcb();
void detect_hands();

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
    detect_hands();

	waitKey(0);

	return 0;
} 

// 피부색 영역 검출
void detect_Ycrcb() {

    Mat ycrcbImg;

    cvtColor(img, ycrcbImg, COLOR_BGR2YCrCb); 
    inRange(ycrcbImg, Scalar(0, 133, 77), Scalar(255, 173, 127), ycrcbImg); 

    skinImg = Mat::zeros(img.size(), CV_8UC3);

    bitwise_and(img, img, skinImg, ycrcbImg);

    imshow("skinImage", skinImg);

    if (skinImg.empty()) {
        cout << "이미지를 읽을 수 없습니다." << endl;
        exit(-1);
    } 
}

void detect_hands() {

    CascadeClassifier hand_classifier("D:\\opencv\\build\\etc\\haarcascades\\hand.xml"); 

    if (hand_classifier.empty()) { 
        cerr << "XML 로드 실패" << endl;
        return;
    }

    vector<Rect> hands; 
    hand_classifier.detectMultiScale(skinImg, hands); 

    Mat handImg = skinImg.clone(); // 이미지 복사 

    for (Rect rc : hands) {
        rectangle(handImg, rc, Scalar(0, 0, 0), 2); 
    }

    imshow("handImage", handImg); 

    if (handImg.empty()) { 
        cout << "이미지를 읽을 수 없습니다." << endl; 
        exit(-1);
    }

    // CascadeClassifier 로는 손 동작을 감지할 수 없음
    // openpose 라이브러리 사용?

}
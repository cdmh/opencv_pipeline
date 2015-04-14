#include "stdafx.h"
#include "opencv_pipeline.h"

namespace cv_pipeline {

namespace tests {

void exhaustive()
{
    using namespace cv_pipeline;

    cv::Mat img;
    img = cv::imread("..\\..\\..\\..\\test data\\caltech\\101_ObjectCategories\\anchor\\image_0008.jpg") | mirror | grey;
    img = load("..\\..\\..\\..\\test data\\images\\Mona_Lisa_headcrop.jpg") | grey | mirror;
    img = "..\\..\\..\\..\\test data\\images\\Mona_Lisa_headcrop.jpg" | verify | grey | mirror;
}

}   // namespace tests

}   // namespace cv_pipeline

int main(int argc, char *argv[])
{
    cv_pipeline::tests::exhaustive();

    cv::Mat img = cv::imread("..\\..\\..\\..\\test data\\images\\Mona_Lisa_headcrop.jpg");
    if (img.empty())
        throw std::runtime_error("file not found");

    cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cvtColor(img, img, cv::COLOR_GRAY2BGR);
    cv::flip(img, img, 1);

	return 0;
}

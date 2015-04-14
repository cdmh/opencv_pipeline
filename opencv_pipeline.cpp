#include "stdafx.h"
#include "opencv_pipeline.h"

namespace cv_pipeline {

namespace tests {

void exhaustive()
{
    using namespace cv_pipeline;

    char const * const test_file = "..\\..\\..\\..\\test data\\images\\Mona_Lisa_headcrop.jpg";

    cv::Mat img = cv::imread(test_file) | mirror | grey;
    img = load(test_file) | grey | mirror;
    img = test_file | verify | grey | mirror;

    test_file | verify;
    test_file | verify | grey | verify | mirror | std::bind(save, "result.png", std::placeholders::_1);
    test_file | verify | grey | mirror | std::bind(save, "result.png", std::placeholders::_1) | verify;
    std::string(test_file) | verify | grey | mirror | std::bind(save, "result.png", std::placeholders::_1) | noverify;

    auto image1 = test_file | noverify | grey;

    auto image = test_file | noverify;
	if (!image.empty())
	    image = image | grey;
}

}   // namespace tests

}   // namespace cv_pipeline

int main(int argc, char *argv[])
{
    cv_pipeline::tests::exhaustive();
	return 0;
}

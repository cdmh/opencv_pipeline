#include "stdafx.h"
#include "opencv_pipeline.h"

namespace cv_pipeline {

namespace tests {

void exhaustive()
{
    using namespace cv_pipeline;

    char const * const test_file = "monalisa.jpg";

    cv::Mat img = cv::imread(test_file) | mirror | grey;
    img = load(test_file) | grey | mirror;
    img = test_file | verify | grey | mirror;

    test_file | verify;
    test_file | verify | grey | verify | mirror | save("result.png");
    test_file | verify | grey | mirror | save("result.png") | verify;
    std::string(test_file) | verify | grey | mirror | save("result.png") | noverify;

    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> mirror_keypoints;
    test_file | verify
        | grey
        | detect("HARRIS", keypoints)
        | extract("SIFT", keypoints)
        | save("result.png") | noverify;
}

}   // namespace tests

}   // namespace cv_pipeline

extern "C" int __stdcall IsDebuggerPresent(void);
int main(int argc, char *argv[])
{
#ifdef _MSC_VER
    // Microsoft Compiler/Debugger help
    if (IsDebuggerPresent())
        cv::setBreakOnError(true);
#endif

    cv::initModule_nonfree();
    cv_pipeline::tests::exhaustive();
	return 0;
}

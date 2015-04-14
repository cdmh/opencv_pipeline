#include "stdafx.h"
#include "opencv_pipeline.h"

namespace { // anonymous namespace

char const * const test_file = "monalisa.jpg";

void pipelines_without_assignment()
{
    using namespace cv_pipeline;

    test_file | verify;
    test_file | verify | grey | verify | mirror | save("result.png");
    test_file | verify | grey | mirror | save("result.png") | verify;
    std::string(test_file) | verify | grey | mirror | save("result.png") | noverify;
}

void reuse_pipeline()
{
    using namespace cv_pipeline;

    auto pipeline = [](
        char const * const filename,
        std::vector<cv::KeyPoint> &keypoints)->cv::Mat {
            return
            filename| verify
                | grey
                | detect("HARRIS", keypoints)
                | extract("SIFT", keypoints);
        };

    std::vector<cv::KeyPoint> keypoints1;
    pipeline("monalisa.jpg", keypoints1)
        | save("monalise-descriptors.jpg") | noverify;

    std::vector<cv::KeyPoint> keypoints2;
    pipeline("da_vinci_human11.jpg", keypoints2)
        | save("da_vinci_human11-descriptors.jpg") | noverify;
}

void exhaustive()
{
    using namespace cv_pipeline;

    // loading an image
    cv::Mat img = cv::imread(test_file) | mirror | grey;
    img = load(test_file) | grey | mirror;
    img = test_file | verify | grey | mirror;

    pipelines_without_assignment();
    reuse_pipeline();
}

}   // anonymous namespace


extern "C" int __stdcall IsDebuggerPresent(void);
int main(int argc, char *argv[])
{
#ifdef _MSC_VER
    // Microsoft Compiler/Debugger help
    if (IsDebuggerPresent())
        cv::setBreakOnError(true);
#endif

    cv::initModule_nonfree();
    exhaustive();
	return 0;
}

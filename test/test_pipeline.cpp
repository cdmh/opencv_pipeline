#include "stdafx.h"
#include "opencv_pipeline.h"

namespace { // anonymous namespace

char const * const test_file = "monalisa.jpg";

void detect_features()
{
    using namespace cv_pipeline;

    std::vector<cv::KeyPoint> keypoints;
    "monalisa.jpg" | verify
        | gray_bgr
        | detect("HARRIS", keypoints)
        | extract("SIFT", keypoints);

    std::vector<std::vector<cv::Point>> regions1;
    auto mscr_sift = 
        "da_vinci_human11.jpg" | verify
        | detect("MSCR", regions1)
        | extract("SIFT", regions1);

    std::vector<std::vector<cv::Point>> regions2;
    auto mser_sift =
        "da_vinci_human11.jpg" | verify
        | detect("MSER", regions2)
        | extract("SIFT", regions2);
}

void pipelines_without_assignment()
{
    using namespace cv_pipeline;

    test_file | verify;
    test_file | verify | gray_bgr | verify | mirror | save("result.png");
    test_file | verify | gray_bgr | mirror | save("result.png") | verify;
    std::string(test_file) | verify | gray_bgr | mirror | save("result.png") | noverify;
}

void reuse_pipeline()
{
    using namespace cv_pipeline;

    auto pipeline = [](char const * const filename)->cv::Mat {
            std::vector<cv::KeyPoint> keypoints;
            return filename| verify | gray_bgr | mirror;
        };

    pipeline("monalisa.jpg")
        | save("monalise-gray_bgr-mirror.png") | noverify;

    pipeline("da_vinci_human11.jpg")
        | save("da_vinci_human11-gray_bgr-mirror.png") | noverify;
}

cv::Mat imshow(const std::string& winname, cv::Mat image)
{
    cv::imshow(winname, image);
    if (cv::waitKey(cvRound(1000.0/25.0)) == 27)
    {
        cvDestroyWindow(winname.c_str());
        throw cv_pipeline::exceptions::end_of_file();
    }
    return image;
}

void play_grey_video()
{
    using namespace cv_pipeline;
    video("../../../../../test data/videos/originals/frame_counter.3gp")
        | gray_bgr
        | mirror
        | std::bind(imshow, "player", std::placeholders::_1)
        | play;
    cvDestroyAllWindows();
}

void exhaustive()
{
    using namespace cv_pipeline;

#if 0
    // test webcam
    camera(0)
        | grey
        | std::bind(imshow, "player", std::placeholders::_1)
        | play;
#endif

    play_grey_video();

    // loading an image
    cv::Mat img = cv::imread(test_file) | mirror;
    img = img | gray_bgr;
    img = load(test_file) | gray_bgr | mirror;
    img = test_file | verify | gray_bgr | mirror;

    pipelines_without_assignment();
    detect_features();
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

#include "stdafx.h"
#include "opencv_pipeline.h"

namespace { // anonymous namespace

#define TESTDATA_DIR "../../../../test data/"
char const * const test_file = TESTDATA_DIR "images/monalisa.jpg";

void detect_features()
{
    using namespace opencv_pipeline;

    std::vector<cv::KeyPoint> keypoints;
    test_file | verify
        | gray_bgr
        | detect("HARRIS", keypoints)
        | extract("SIFT", keypoints);

    std::vector<std::vector<cv::Point>> regions1;
    auto mscr_sift = 
        test_file | verify
        | detect("MSCR", regions1)
        | extract("SIFT", regions1);

    std::vector<std::vector<cv::Point>> regions2;
    auto mser_sift =
        test_file | verify
        | detect("MSER", regions2)
        | extract("SIFT", regions2);
}

void pipelines_without_assignment()
{
    using namespace opencv_pipeline;

    test_file | verify;
    test_file | verify | gray_bgr | verify | mirror | save("result.png");
    test_file | verify | gray_bgr | mirror | save("result.png") | verify;
    std::string(test_file) | verify | gray_bgr | mirror | save("result.png") | noverify;
}

void reuse_pipeline()
{
    using namespace opencv_pipeline;

    auto pipeline = [](char const * const filename)->cv::Mat {
            std::vector<cv::KeyPoint> keypoints;
            return filename| verify | gray_bgr | mirror;
        };

    pipeline(test_file)
        | save("monalisa-gray_bgr-mirror.png") | noverify;

    pipeline(TESTDATA_DIR "images/da_vinci_human11.jpg")
        | save("da_vinci_human11-gray_bgr-mirror.png") | verify;
}

cv::Mat imshow(char const * const winname, cv::Mat const &image)
{
    cv::imshow(winname, image);
    if (cv::waitKey(cvRound(1000.0/25.0)) == 27)
    {
        cvDestroyWindow(winname);
        throw opencv_pipeline::exceptions::end_of_file();
    }
    return image;
}

void play_grey_video()
{
    using namespace opencv_pipeline;
    auto vid = video(TESTDATA_DIR "videos/originals/frame_counter.3gp");
    auto show = std::bind(imshow, "player", std::placeholders::_1);
    vid | gray_bgr
        | mirror
        | show
        | play;
    cvDestroyAllWindows();
}


// http://rnd.azoft.com/instant-license-plate-recognition-in-ios-apps/
cv::Mat preprocess_license_plate(cv::Mat const &src)
{
    using namespace opencv_pipeline;
    return src
         | dilate(3, 9)                          // close & subtract
         | erode(3, 9)
         | subtract(src)
         | cv::abs
         | sobel(1, 0, 3)                        // edges & blur
         | gaussian_blur(5, 5)
         | dilate(3, 9)                          // closing
         | erode(3, 9)
         | convert(CV_8UC1)
         | threshold(0., 255.);                  // create binary image
}

void license_plate()
{
    using namespace opencv_pipeline;

    char const * const filename = TESTDATA_DIR "images/vehicle-license-plate-recognition-algorithm-02.jpg";

    cv::Mat src = filename | verify | if_(channels(3), gray);// | convert(CV_64FC1);

    preprocess_license_plate(src);
    cv::Mat mask = preprocess_license_plate(src);

    cv::Mat overlay;
    cv::merge(
        { cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1),
          cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1),
          mask
        },
        overlay);

    double alpha = 0.6;
    cv::Mat dst;
    addWeighted(src | color_space(cv::COLOR_GRAY2BGR), alpha, overlay, 1.-alpha, 0.0, dst);


// step1
    std::vector<std::vector<cv::Point>> contours;  
    findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);  

// step2
    std::vector<cv::RotatedRect> rects;
    for (auto itc = contours.cbegin(); itc != contours.cend(); )
    {
        cv::RotatedRect mr = minAreaRect(cv::Mat(*itc));

        auto const area   = fabs(contourArea(*itc));
        auto const bbArea = mr.size.width * mr.size.height;
        auto const ratio  = area/bbArea;
                
        if (ratio < 0.45  ||  bbArea < 400)
            itc = contours.erase(itc);
        else
        {
            ++itc;
            rects.push_back(mr);
        }
    }


    auto result = src  | color_space(cv::COLOR_GRAY2BGR);
    for (auto &rect : rects)
    {
        rect.size.width  += rect.size.width  * 0.15126f;
        rect.size.height += rect.size.height * 0.625f;

        // step 7
        std::vector< std::vector<cv::Point> > contours;  
        findContours(src(rect.boundingRect() & roi(src)) | threshold(0., 255.), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

        double largest = 0.;
        size_t index = std::numeric_limits<size_t>::max();
        for (size_t i=0; i<contours.size(); ++i)
        {
            auto area = contourArea(contours[i], false);
            if (area > largest)
            {
                largest = area;
                index = i;
            }
        }

        if (index != std::numeric_limits<size_t>::max())
        {
            cv::RotatedRect box = minAreaRect(cv::Mat(contours[index]));

            double angle = box.angle;
            if (angle < -45.)
                angle += 90.;

            cv::Point2f vertices[4];
            box.points(vertices);

            cv::Mat area = result(rect.boundingRect() & roi(result));
            for (int i = 0; i < 4; ++i)
            {
                line(area, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 255, 0));
            }
        }
    }
}

void exhaustive()
{
    using namespace opencv_pipeline;

    try
    {
        license_plate();
    }
    catch (exceptions::file_not_found const &e)
    {
        std::cerr << "File not found: " << e.what() << '\n';
    }

#if 0
    // test webcam
    camera(0)
        | grey
        | std::bind(imshow, "player", std::placeholders::_1)
        | play;
#endif

    // loading an image
    cv::Mat img = cv::imread(test_file) | verify | mirror;
    img = img | gray_bgr;
    img = load(test_file) | gray_bgr | mirror;
    img = test_file | verify | gray_bgr | mirror;

    pipelines_without_assignment();
    detect_features();
    reuse_pipeline();

    play_grey_video();
}

}   // anonymous namespace


extern "C" int __stdcall IsDebuggerPresent(void);
int main(int /*argc*/, char * /*argv*/[])
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

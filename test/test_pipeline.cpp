#include "stdafx.h"
#include "opencv_pipeline.h"

namespace { // anonymous namespace

#define TESTDATA_DIR "D:/QMUL/test data/"
std::filesystem::path test_file = TESTDATA_DIR "images/monalisa.jpg";

void detect_features()
{
    using namespace opencv_pipeline;

    // test keypoint feature detections and descriptor extraction
    {
        // extract SIFT descriptors from HARRIS features (features)
        auto descriptors1 = test_file | verify
            | gray_bgr
            | keypoints("HARRIS")
            | descriptors("SIFT")
            | end;                  // optional end video_pipeline_terminator
        static_assert(std::is_same<cv::Mat, decltype(descriptors1)>::value);

        // find HARRIS keypoint features
        auto img = test_file | verify | gray_bgr;
        auto kps = img | keypoints("HARRIS") | end;
        static_assert(std::is_same<std::vector<cv::KeyPoint>, decltype(kps)>::value);

        // continuation of the pipeline to extract descriptors; the order of
        // keypoint and image is unimportant
        auto dsc1 = img | kps | descriptors("SIFT");
        auto dsc2 = kps | img | descriptors("SIFT");
        static_assert(std::is_same<cv::Mat, decltype(dsc1)>::value);
        static_assert(std::is_same<cv::Mat, decltype(dsc2)>::value);
    }

    // test region feature detections and descriptor extraction
    {
        // extract MSCR descriptors from SIFT features (features)
        auto descriptors1 = test_file | verify
            | gray_bgr
            | regions("MSCR")
            | descriptors("SIFT")
            | end;                  // optional end video_pipeline_terminator
        static_assert(std::is_same<cv::Mat, decltype(descriptors1)>::value);

        // find MSCR region features
        auto img = test_file | verify;
        auto rgn = img | regions("MSCR") | end;
        static_assert(std::is_same<std::vector<std::vector<cv::Point>>, decltype(rgn)>::value);

        // continuation of the pipeline to extract descriptors; the order of
        // keypoint and image is unimportant
        auto mscr_sift1 = img | rgn | descriptors("SIFT");
        auto mscr_sift2 = rgn | img | descriptors("SIFT");
        static_assert(std::is_same<cv::Mat, decltype(mscr_sift1)>::value);
        static_assert(std::is_same<cv::Mat, decltype(mscr_sift2)>::value);

        std::vector<std::vector<cv::Point>> regions2;
        auto mser_sift =
            test_file | verify
            | regions("MSER")
            | descriptors("SIFT");
        static_assert(std::is_same<cv::Mat, decltype(mser_sift)>::value);
    }
}

void file_processing()
{
    using namespace opencv_pipeline;
    auto pngs = directory_iterator(TESTDATA_DIR "images/*.png");
    auto show_image = pipeline | show("Image") | waitkey(0);
    auto images = pngs | show_image;
    static_assert(std::is_same<std::vector<std::filesystem::path>, decltype(pngs)>::value);
    static_assert(std::is_same<opencv_pipeline::persistent_pipeline, decltype(show_image)>::value);
    static_assert(std::is_same<std::vector<cv::Mat>, decltype(images)>::value);

    images = directory_iterator(TESTDATA_DIR "images/*.png")
      | (foreach | gray | mirror | show("Image") | waitkey(0));
}

void list_processing()
{
    using namespace opencv_pipeline;
    using namespace opencv_pipeline::array;

    auto pipeline = foreach | gray | sobel(5, 5, 7) | show("Image") | waitkey(0);
    {
        auto files = (std::filesystem::path(TESTDATA_DIR "images/african-art-1732250_960_720.jpg"),
                      std::filesystem::path(TESTDATA_DIR "images/rgb.png"),
                      std::filesystem::path(TESTDATA_DIR "images/da_vinci_human11.jpg"));
        static_assert(std::is_same<std::array<std::filesystem::path, 3>, decltype(files)>::value);

        auto images = files | pipeline;
        static_assert(std::is_same<std::array<cv::Mat, 3>, decltype(images)>::value);
    }

    {
        auto images = ((std::filesystem::path(TESTDATA_DIR "images/african-art-1732250_960_720.jpg") | verify | pipeline),
                       (std::filesystem::path(TESTDATA_DIR "images/rgb.png")                         | verify | pipeline),
                       (std::filesystem::path(TESTDATA_DIR "images/da_vinci_human11.jpg"))           | verify | pipeline);
        static_assert(std::is_same<std::array<cv::Mat, 3>, decltype(images)>::value);
    }
}

void pipelines_without_assignment()
{
    using namespace opencv_pipeline;

    test_file | verify;
    test_file | verify | gray_bgr | verify | mirror | save("result1.png");
    test_file | verify | gray_bgr | mirror | save("result2.png") | verify;
}

void reuse_pipeline()
{
    using namespace opencv_pipeline;

    auto pipeline = [](std::filesystem::path pathname)->cv::Mat {
            return pathname | verify | gray_bgr | mirror;
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
         | threshold(0., 255.);                  // pipeline binary image
}

void license_plate()
{
    using namespace opencv_pipeline;

    char const * const filename = TESTDATA_DIR "images/vehicle-license-plate-recognition-algorithm-02.jpg";

    auto src = filename | verify | if_(channels(3), gray);
    auto mask = preprocess_license_plate(src);

    cv::Mat overlay;
    cv::merge(
        cv::InputArrayOfArrays(
            std::vector<cv::Mat>{
                cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1),  // Blue channel
                cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1),  // Green channel
                mask                                            // Red channel
            }
        ),
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
        contours.clear();
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

    test_file | verify   | show("")        | waitkey(0);
    test_file | noverify | show("window1") | waitkey(0);

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
    auto img = cv::imread(test_file.u8string()) | verify | mirror;
    img = img | gray_bgr;
    img = load(test_file) | gray_bgr | mirror;
    img = test_file | verify | gray_bgr | mirror;
    static_assert(std::is_same<cv::Mat, decltype(img)>::value);

    list_processing();
    file_processing();
    pipelines_without_assignment();
    detect_features();
    reuse_pipeline();

    play_grey_video();
}

}   // anonymous namespace

namespace examples {
void run();
}   // namespace examples


extern "C" int __stdcall IsDebuggerPresent(void);
int main(int /*argc*/, char * /*argv*/[])
{
#ifdef _MSC_VER
    // Microsoft Compiler/Debugger help
    if (IsDebuggerPresent())
        cv::setBreakOnError(true);
#endif

#if CV_MAJOR_VERSION==2
    cv::initModule_nonfree();
#endif

    exhaustive();
    examples::run();
	return 0;
}

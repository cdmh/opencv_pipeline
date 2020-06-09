#include "stdafx.h"
#include "opencv_pipeline.h"

namespace examples {

#define TESTDATA_DIR "D:/QMUL/test data/"
std::filesystem::path test_file = TESTDATA_DIR "images/monalisa.jpg";

void extract_descriptors_from_keypoints()
{
    using namespace opencv_pipeline;
    TESTDATA_DIR "images/monalisa.jpg" | load
        | gray
        | keypoints("HARRIS") | descriptors("SIFT")
        | save("harris_sift.png") | load_ignore_failure;
}

void extract_descriptors_from_regions()
{
    using namespace opencv_pipeline;
    TESTDATA_DIR "images/monalisa.jpg" | load
        | regions("MSCR") | descriptors("SIFT")
        | save("mscr_sift.png") | load_ignore_failure;
}

void reuse_pipeline()
{
    using namespace opencv_pipeline;
    auto show_grey_mirrored = pipeline | gray | mirror | show("Image") | waitkey(0);
    TESTDATA_DIR "images/monalisa.jpg"                    | load | show_grey_mirrored;
    TESTDATA_DIR "images/african-art-1732250_960_720.jpg" | load | show_grey_mirrored;
}

void parameterised_pipeline()
{
    using namespace opencv_pipeline;
    auto pipeline = [](char const * const filename)->cv::Mat {
        return
            filename| load
                | gray
                | keypoints("HARRIS")
                | descriptors("SIFT");
    };

    pipeline(TESTDATA_DIR "images/monalisa.jpg")
        | save("monalisa-harris-sift.jpg") | load_ignore_failure;

    pipeline(TESTDATA_DIR "images/da_vinci_human11.jpg")
        | save("da_vinci_human11-harris-sift.png") | load_ignore_failure;
}

void run()
{
    extract_descriptors_from_keypoints();
    extract_descriptors_from_regions();
    reuse_pipeline();
    parameterised_pipeline();
}

}   // namespace examples

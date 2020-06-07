#include "stdafx.h"
#include "opencv_pipeline.h"

namespace examples {

#define TESTDATA_DIR "D:/QMUL/test data/"
std::filesystem::path test_file = TESTDATA_DIR "images/monalisa.jpg";

void extract_descriptors_from_keypoints()
{
    using namespace opencv_pipeline;
    TESTDATA_DIR "images/monalisa.jpg" | verify
        | gray
        | keypoints("HARRIS") | descriptors("SIFT")
        | save("harris_sift.png") | noverify;
}

void extract_descriptors_from_regions()
{
    using namespace opencv_pipeline;
    TESTDATA_DIR "images/monalisa.jpg" | verify
        | regions("MSCR") | descriptors("SIFT")
        | save("mscr_sift.png") | noverify;
}

void reuse_pipeline()
{
    using namespace opencv_pipeline;
    auto pipeline = delay | gray | mirror | show("Image") | waitkey(0);
    TESTDATA_DIR "images/monalisa.jpg"                    | verify | pipeline;
    TESTDATA_DIR "images/african-art-1732250_960_720.jpg" | verify | pipeline;
}

void parameterised_pipeline()
{
    using namespace opencv_pipeline;
    auto pipeline = [](char const * const filename)->cv::Mat {
        return
            filename| verify
                | gray
                | keypoints("HARRIS")
                | descriptors("SIFT");
    };

    pipeline(TESTDATA_DIR "images/monalisa.jpg")
        | save("monalisa-harris-sift.jpg") | noverify;

    pipeline(TESTDATA_DIR "images/da_vinci_human11.jpg")
        | save("da_vinci_human11-harris-sift.png") | noverify;
}

void run()
{
    extract_descriptors_from_keypoints();
    extract_descriptors_from_regions();
    reuse_pipeline();
    parameterised_pipeline();
}

}   // namespace examples

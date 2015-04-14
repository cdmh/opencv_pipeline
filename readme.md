OpenCV Pipeline
===============

A C++11 pipeline interface to OpenCV 2.4. See `develop` branch for current implementation.

Why write this:

    cv::Mat img = cv::imread("..\\..\\..\\..\\test data\\images\\Mona_Lisa_headcrop.jpg");
    if (img.empty())
        throw std::runtime_error("file not found");

    cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cvtColor(img, img, cv::COLOR_GRAY2BGR);
    cv::flip(img, img, 1);

when you can write this:

    using namespace cv_pipeline;
    cv::Mat img = "monalisa.jpg" | verify | grey | mirror;

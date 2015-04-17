#pragma once

namespace cv_pipeline {

inline
cv::Mat load(char const * const pathname)
{
    return cv::imread(pathname);
}

inline
cv::Mat load(std::string const &pathname)
{
    return cv::imread(pathname);
}

inline
std::function<cv::Mat (cv::Mat const &)>
save(char const * const pathname)
{
    return std::bind(detail::save, pathname, std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<cv::KeyPoint> &keypoints)
{
    return std::bind(detail::detect_keypoints, detector, std::ref(keypoints), std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
detect(char const * const detector, std::vector<std::vector<cv::Point>> &regions)
{
    return std::bind(detail::detect_regions, detector, std::ref(regions), std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
extract(char const * const detector, std::vector<cv::KeyPoint> &keypoints)
{
    return std::bind(detail::extract_keypoints, detector, std::ref(keypoints), std::placeholders::_1);
}

inline
std::function<cv::Mat (cv::Mat const &)>
extract(char const * const detector, std::vector<std::vector<cv::Point>> &regions)
{
    return std::bind(detail::extract_regions, detector, std::ref(regions), std::placeholders::_1);
}

//
// image manipulation
//

inline
cv::Mat gray(cv::Mat image)
{
    cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cvtColor(image, image, cv::COLOR_GRAY2BGR);
    return image;
}

inline
cv::Mat grey(cv::Mat image)
{
    return gray(image);
}

inline
cv::Mat mirror(cv::Mat image)
{
    flip(image, image, 1);
    return image;
}

//
// operators -- these are not forward referenced in the
// header file as they are the mechanics used implicitly
// by the user rather than explicitly called functions
//

// apply a function to an image
inline
cv::Mat operator|(cv::Mat const &left, cv::Mat(*right)(cv::Mat const &))
{
    return right(left);
}

// apply a function to an image - enables std::bind() bound parameters
inline
cv::Mat operator|(cv::Mat const &left, std::function<cv::Mat(cv::Mat const &)> right)
{
    return right(left);
}

// load an image with optional verification.
// whether or not to verify is compulsory -- verify or noverify must
// go between a load and any subsequent manipulations through the
// pipeline interface
inline
cv::Mat operator|(char const * const pathname, verify_result verify)
{
    cv::Mat image = load(pathname);
    if (verify  &&  image.empty())
        throw exceptions::file_not_found(pathname);
    return image;
}

inline
cv::Mat operator|(std::string const pathname, verify_result verify)
{
    cv::Mat image = load(pathname);
    if (verify  &&  image.empty())
        throw exceptions::file_not_found(pathname.c_str());
    return image;
}

// verify an image is not empty
inline
cv::Mat operator|(cv::Mat const &image, verify_result verify)
{
    if (verify  &&  image.empty())
        throw exceptions::bad_image();
    return image;
}



class video_pipeline
{
  public:
    video_pipeline(char const * const pathname)
    {
        capture_.open(pathname);
    }

    cv::Mat run()
    {
        cv::Mat image;
        capture_ >> image;
        if (image.empty())
            throw exceptions::end_of_file();
        return image;
    }

    video_pipeline(video_pipeline const &)           = delete;
    video_pipeline &operator=(video_pipeline const &) = delete;

    cv::VideoCapture &capture() { return capture_; }

  private:
    cv::VideoCapture capture_;
};

inline
video_pipeline
video(char const * const pathname)
{
    return video_pipeline(pathname);
}

template<typename LHS, typename RHS>
class chain_link
{
  public:
    chain_link(LHS &left, RHS &right) : lhs(left), rhs(right)
    {
    }

    cv::Mat run()
    {
        return rhs(lhs.run());
    }

  private:
    LHS lhs;
    RHS rhs;
};

chain_link<
    video_pipeline &,
    std::function<cv::Mat (cv::Mat)>>
operator|(
    video_pipeline &lhs,
    std::function<cv::Mat (cv::Mat)> rhs)
{
    return {lhs,rhs};
}

template<typename LHS, typename RHS>
chain_link<
    chain_link<LHS, RHS>,
    std::function<cv::Mat (cv::Mat)>>
operator|(
    chain_link<LHS, RHS>             lhs, 
    std::function<cv::Mat (cv::Mat)> rhs)
{
    return {lhs, rhs};
}

typedef
enum { play }
terminator;

template<typename LHS, typename RHS>
bool const
operator|(chain_link<LHS, RHS> lhs, terminator)
{
    try
    {
        while (1)
            lhs.run();
    }
    catch (exceptions::end_of_file &)
    {
        return true;
    }
    return false;
}

}   // namespace cv_pipeline

#pragma once

namespace opencv_pipeline {

inline
cv::Mat load_image(std::filesystem::path pathname, int flags=cv::IMREAD_COLOR)
{
    return cv::imread(pathname.u8string(), flags);
}

inline
pipeline_fn_t
save(std::filesystem::path pathname)
{
    return std::bind(detail::save, std::placeholders::_1, pathname);
}

inline
pipeline_fn_t
show(char const * const window_name)
{
    return std::bind(detail::show, window_name, std::placeholders::_1);
}

inline
cv::Rect roi(cv::Mat const &image)
{
    cv::Point tl;
    cv::Size  size;
    image.locateROI(size, tl);
    return cv::Rect(tl, size);
}

namespace detail {

// T = cv::KeyPoint           for points
// T = std::vector<cv::Point> for regions
template<typename T>
struct feature_detector
{
    feature_detector(std::string name) : name(name)
    {
    }

    feature_detector(cv::Mat img, std::vector<T> kps) : image(img), features(kps)
    {
    }

    cv::Mat operator()(cv::Mat const &img);

    operator std::vector<T>()
    {
        return features;
    }
    
    cv::Mat        image;
    std::string    name;
    std::vector<T> features;
};

template<>
inline
cv::Mat feature_detector<cv::KeyPoint>::operator()(cv::Mat const &img)
{
    image = img;
    return detail::detect_keypoints(name, features, image);
}

template<>
inline
cv::Mat feature_detector<std::vector<cv::Point>>::operator()(cv::Mat const &img)
{
    image = img;
    return detail::detect_regions(name, features, image);
}

struct feature_extractor
{
    feature_extractor(std::string name) : name(name)
    {
    }

    std::string name;
};

}   // namespace detail

inline
detail::feature_detector<cv::KeyPoint>
keypoints(std::string detector)
{
    return detail::feature_detector<cv::KeyPoint>(std::move(detector));
}

inline
detail::feature_extractor
descriptors(std::string extractor)
{
    return detail::feature_extractor(extractor);
}

inline
detail::feature_detector<cv::KeyPoint> &&operator|(cv::Mat image, detail::feature_detector<cv::KeyPoint> &&detector)
{
    detector(image);
    return std::move(detector);
}

inline
cv::Mat operator|(detail::feature_detector<cv::KeyPoint> const &detector, detail::feature_extractor const &extractor)
{
    return detail::extract_keypoints(extractor.name, detector.features, detector.image);
}

inline
detail::feature_detector<cv::KeyPoint> operator|(cv::Mat image, std::vector<cv::KeyPoint> const &keypoints)
{
    return detail::feature_detector<cv::KeyPoint>(image, keypoints);
}

inline
detail::feature_detector<cv::KeyPoint> operator|(std::vector<cv::KeyPoint> const &keypoints, cv::Mat image)
{
    return detail::feature_detector<cv::KeyPoint>(image, keypoints);
}

// enable early pipeline to detect features without extracting descriptors
// e.g. auto kps = test_file | load | gray_bgr | features("HARRIS") | end;
//      static_assert(std::is_same<std::vector<cv::KeyPoint>, decltype(kps)>::value);
inline
std::vector<cv::KeyPoint> operator|(detail::feature_detector<cv::KeyPoint> const &detector, pipeline_terminator)
{
    return detector.features;
}

inline
detail::feature_detector<std::vector<cv::Point>>
regions(std::string detector)
{
    return detail::feature_detector<std::vector<cv::Point>>(std::move(detector));
}

inline
detail::feature_detector<std::vector<cv::Point>> &&operator|(cv::Mat image, detail::feature_detector<std::vector<cv::Point>> &&detector)
{
    detector(image);
    return std::move(detector);
}

inline
cv::Mat operator|(detail::feature_detector<std::vector<cv::Point>> const &detector, detail::feature_extractor const &extractor)
{
    return detail::extract_regions(extractor.name, detector.features, detector.image);
}

inline
detail::feature_detector<std::vector<cv::Point>> operator|(cv::Mat image, std::vector<std::vector<cv::Point>> const &features)
{
    return detail::feature_detector<std::vector<cv::Point>>(image, features);
}

inline
detail::feature_detector<std::vector<cv::Point>> operator|(std::vector<std::vector<cv::Point>> const &features, cv::Mat image)
{
    return detail::feature_detector<std::vector<cv::Point>>(image, features);
}

// enable early pipeline to detect features without extracting descriptors
// e.g. auto kps = test_file | load | gray_bgr | features("HARRIS") | end;
//      static_assert(std::is_same<std::vector<std::vector<cv::KeyPoint>>, decltype(kps)>::value);
inline
std::vector<std::vector<cv::Point>> operator|(detail::feature_detector<std::vector<cv::Point>> const &detector, pipeline_terminator)
{
    return detector.features;
}

//
// operators -- these are not forward referenced in the
// header file as they are the mechanics used implicitly
// by the user rather than explicitly called functions
//

//
// image pipeline
//

// apply a function to an image
inline
cv::Mat operator|(cv::Mat const &left, cv::Mat(*right)(cv::Mat const &))
{
    return right(left);
}

// apply a function to an image - enables std::bind() bound parameters
inline
cv::Mat operator|(cv::Mat const &left, pipeline_fn_t const &right)
{
    return right(left);
}

// some OpenCV functions return MatExpr
inline
cv::Mat operator|(cv::Mat const &left, cv::MatExpr(*right)(cv::Mat const &))
{
    return right(left);
}

//
// image manipulation
//

inline
pipeline_fn_t
color_space(int code)
{
    using namespace std::placeholders;
    return std::bind(detail::color_space, _1, code);
}

inline
pipeline_fn_t
convert(int type, double alpha=1.0, double beta=0.0)
{
    using namespace std::placeholders;
    return std::bind(detail::convert, _1, type, alpha, beta);
}

inline
pipeline_fn_t
dilate(int dx, int dy)
{
    using namespace std::placeholders;
    return std::bind(detail::dilate, _1, dx, dy);
}

inline
pipeline_fn_t
erode(int dx, int dy)
{
    using namespace std::placeholders;
    return std::bind(detail::erode, _1, dx, dy);
}

inline
pipeline_fn_t
gaussian_blur(int dx, int dy, double sigmaX=0.0, double sigmaY=0.0, int border=cv::BORDER_DEFAULT)
{
    using namespace std::placeholders;
    return std::bind(detail::gaussian_blur, _1, dx, dy, sigmaX, sigmaY, border);
}

inline
cv::Mat clone(cv::Mat const &image)
{
    return image.clone();
}

inline
cv::Mat equalizeHist(cv::Mat const &image)
{
    cv::Mat dst;
    cv::equalizeHist(image, dst);
    return dst;
}

inline
cv::Mat gray(cv::Mat const &image)
{
    return image | color_space(cv::COLOR_BGR2GRAY);
}

inline
cv::Mat gray_bgr(cv::Mat const &image)
{
    return image | gray | color_space(cv::COLOR_GRAY2BGR);
}

inline
cv::Mat mirror(cv::Mat const &image)
{
    cv::Mat dst;
    flip(image, dst, 1);
    return dst;
}

inline
pipeline_fn_t
resize(double fx, double fy, int interpolation)
{
    auto resizer = [fx, fy, interpolation](cv::Mat const &src) -> cv::Mat {
        cv::Mat dst;
        cv::resize(src, dst, cv::Size(), fx, fy, interpolation);
        return dst;
    };

    using namespace std::placeholders;
    return std::bind(resizer, _1);
}

inline
pipeline_fn_t
sobel(int dx, int dy, int ksize=3, double scale=1, double delta=0, int border=cv::BORDER_DEFAULT)
{
    using namespace std::placeholders;
    return std::bind(detail::sobel, _1, dx, dy, ksize, scale, delta, border);
}

inline
pipeline_fn_t
subtract(cv::Mat const &other)
{
    using namespace std::placeholders;
    return std::bind(detail::subtract, _1, other);
}

inline
pipeline_fn_t
threshold(double thresh, double maxval, int type=CV_THRESH_BINARY | CV_THRESH_OTSU)
{
    using namespace std::placeholders;
    return std::bind(detail::threshold, _1, thresh, maxval, type);
}


//
// conditions
//


inline
cv::Mat noop(cv::Mat const &image)
{
    return image;
}

inline
cv::Mat verify(cv::Mat const &image)
{
    if (image.empty())
        throw exceptions::bad_image();
    return image;
}


inline
pipeline_fn_t
if_(std::function<bool const (cv::Mat const &)> cond, pipeline_fn_t fn)
{
    using namespace std::placeholders;
    return std::bind(detail::if_, _1, cond, fn);
}

inline
pipeline_fn_t
if_(bool const cond, pipeline_fn_t fn)
{
    return cond? fn : noop;
}

inline
persistent_pipeline
if_(bool const cond, persistent_pipeline pipeline)
{
    return cond? pipeline : persistent_pipeline();
}


//
// image attributes
//

inline
std::function<bool const (cv::Mat const &)>
channels(int num)
{
    using namespace std::placeholders;
    return std::bind(detail::channels, _1, num);
}




class video_pipeline
{
  public:
    video_pipeline(video_pipeline &&) = default;

    video_pipeline(int device)
    {
        if (!capture_.open(device))
            last_error_ = "Unable to open camera #" + std::to_string(device);
    }

    video_pipeline(std::filesystem::path pathname)
    {
       if (!capture_.open(pathname.u8string()))
            last_error_ = "Unable to open video file: " + pathname.u8string();
    }

    std::string last_error() const
    {
        return last_error_;
    }

    bool open() const
    {
        return capture_.isOpened();
    }

    cv::Mat next_frame()
    {
        cv::Mat image;
        capture_ >> image;
        if (image.empty())
            throw exceptions::end_of_file();
        return image;
    }

    video_pipeline()                                  = delete;
    video_pipeline(video_pipeline const &)            = delete;
    video_pipeline &operator=(video_pipeline &&)      = delete;
    video_pipeline &operator=(video_pipeline const &) = delete;

  private:
    cv::VideoCapture capture_;
    std::string      last_error_;
};

// capture video from a file
inline
video_pipeline
video(std::filesystem::path pathname)
{
    return video_pipeline(pathname);
}

// capture video from a camera device
inline
video_pipeline
camera(int device)
{
    return video_pipeline(device);
}


//
// video pipeline
//

inline
std::pair<video_pipeline &, pipeline_fn_t>
operator|(
    video_pipeline &lhs,
    pipeline_fn_t const &rhs)
{
    return {lhs,rhs};
}

inline
std::pair<video_pipeline &, pipeline_fn_t>
operator|(video_pipeline &lhs, cv::Mat (*rhs)(cv::Mat const &))
{
    return {lhs,rhs};
}

// pipe directly into a function
template<typename LHS, typename RHS>
std::pair<std::pair<LHS, RHS>, pipeline_fn_t>
operator|(std::pair<LHS, RHS> lhs, cv::Mat(*rhs)(cv::Mat const &))
{
    return {lhs, rhs};
}

// pipe into a bound function (std::bind)
template<typename LHS, typename RHS>
std::pair<std::pair<LHS, RHS>, pipeline_fn_t>
operator|(std::pair<LHS, RHS> lhs, pipeline_fn_t const &rhs)
{
    return {lhs, rhs};
}

typedef
enum { play }
video_pipeline_terminator;

inline
cv::Mat next_frame(video_pipeline &pipeline)
{
    return pipeline.next_frame();
}

template<typename LHS, typename RHS>
cv::Mat next_frame(std::pair<LHS, RHS> chain)
{
    return chain.second(next_frame(chain.first));
}

#pragma warning(push)
#pragma warning(disable: 4127)  // C4127 conditional expression is constant
template<typename LHS, typename RHS>
bool const
operator|(std::pair<LHS, RHS> lhs, video_pipeline_terminator)
{
    try
    {
        while (1)
            next_frame(lhs);
    }
    catch (exceptions::end_of_file &)
    {
        return true;
    }
    return false;
}
#pragma warning(pop)


// load an image with optional verification.
// whether or not to load is compulsory -- load or load_ignore_failure must
// go between a load and any subsequent manipulations through the
// pipeline interface
inline
cv::Mat operator|(std::filesystem::path pathname, image_loader verify)
{
    cv::Mat image = load_image(pathname);
    if (verify != load_ignore_failure  &&  image.empty())
        throw exceptions::file_not_found(pathname);
    return image;
}

// load an image is not empty
inline
cv::Mat operator|(cv::Mat const &image, image_loader verify)
{
    if (verify != load_ignore_failure  &&  image.empty())
        throw exceptions::bad_image();
    return image;
}

// load a video capture is open
inline
video_pipeline const &operator|(video_pipeline const &pipeline, image_loader verify)
{
    if (verify != load_ignore_failure  &&  !pipeline.open())
        throw exceptions::bad_video(pipeline.last_error());
    return pipeline;
}

}   // namespace opencv_pipeline

#pragma once

namespace opencv_pipeline {

typedef
enum { pipeline=1, foreach }
delay_result;

inline
persistent_pipeline::persistent_pipeline(std::function<cv::Mat (cv::Mat const &)> &&fn)
{
    fn_.push_back(std::forward<std::function<cv::Mat (cv::Mat const &)>>(fn));
}

inline
persistent_pipeline &persistent_pipeline::append(std::function<cv::Mat (cv::Mat const &)> &&fn)
{
    fn_.push_back(std::forward<std::function<cv::Mat (cv::Mat const &)>>(fn));
    return *this;
}

inline
cv::Mat persistent_pipeline::operator()(cv::Mat &&image) const
{
    for (auto &fn : fn_)
        image = fn(image);
    return image;
}

// pipeline a persistent pipeline
inline
persistent_pipeline operator|(delay_result, std::function<cv::Mat (cv::Mat const &)> rhs)
{
    return persistent_pipeline(std::move(rhs));
}

inline
persistent_pipeline operator|(delay_result, cv::Mat (*rhs)(cv::Mat const &))
{
    return persistent_pipeline(rhs);
}

// run a persistent_pipeline
inline
cv::Mat operator|(cv::Mat lhs, persistent_pipeline const &rhs)
{
    return rhs(std::move(lhs));
}

// construct pipes
inline
persistent_pipeline operator|(persistent_pipeline lhs, std::function<cv::Mat (cv::Mat const &)> rhs)
{
    return lhs.append(std::move(rhs));
}

inline
persistent_pipeline operator|(std::function<cv::Mat (cv::Mat const &)> lhs, persistent_pipeline rhs)
{
    return persistent_pipeline(std::move(lhs));
}

inline
persistent_pipeline operator|(persistent_pipeline lhs, cv::Mat (*rhs)(cv::Mat const &))
{
    return lhs.append(rhs);
}

inline
persistent_pipeline operator|(cv::Mat (*lhs)(cv::Mat const &), persistent_pipeline rhs)
{
    return rhs.append(lhs);
}


// get a list files in a directory that match a wildcard
inline
std::vector<std::filesystem::path>
directory_iterator(std::filesystem::path pathname)
{
    std::vector<cv::String> results;
    cv::glob(pathname.u8string(), results, false);

    std::vector<std::filesystem::path> pathnames;
    for (auto const &result : results)
        pathnames.emplace_back(result.c_str());
    return pathnames;
}

inline
std::vector<cv::Mat>
operator|(std::vector<std::filesystem::path> const &pathnames,
          persistent_pipeline        const &pipeline)
{
    std::vector<cv::Mat> processed;
    for (auto const &pathname : pathnames)
        processed.emplace_back(pathname | verify | pipeline);
    return processed;
}


}   // namespace opencv_pipeline

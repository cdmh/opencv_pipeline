#pragma once

namespace opencv_pipeline {

typedef
enum { pipeline=1, foreach }
delay_result;

inline
persistent_pipeline::persistent_pipeline(pipeline_fn_t &&fn)
{
    fn_.push_back(std::forward<pipeline_fn_t>(fn));
}

inline
persistent_pipeline &persistent_pipeline::append(pipeline_fn_t &&fn)
{
    fn_.push_back(std::forward<pipeline_fn_t>(fn));
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
persistent_pipeline operator|(delay_result, pipeline_fn_t rhs)
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
persistent_pipeline operator|(persistent_pipeline lhs, pipeline_fn_t rhs)
{
    return lhs.append(std::move(rhs));
}

inline
persistent_pipeline operator|(pipeline_fn_t lhs, persistent_pipeline rhs)
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

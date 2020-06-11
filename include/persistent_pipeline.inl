#pragma once

namespace opencv_pipeline {

typedef
enum { pipeline=1, apply }
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

template<typename C>
inline
pipeline_fn_t
foreach(C const &container, persistent_pipeline rhs)
{
    return std::bind(
        [](cv::Mat const &image, auto const &container, persistent_pipeline rhs) -> cv::Mat {
            int index = 0;
            cv::Mat result = image;
            for (auto const &value : container)
                result = rhs(result, index, value);
            return result;
        },
        std::placeholders::_1, std::cref(container), rhs);
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
          persistent_pipeline                const &pipeline)
{
    std::vector<cv::Mat> results;
    for (auto const &pathname : pathnames)
        results.emplace_back(pathname | load | pipeline);
    return results;
}

template<size_t N>
std::array<cv::Mat, N>
operator|(std::array<std::filesystem::path, N> const &pathnames,
          persistent_pipeline                  const &pipeline)
{
    std::array<cv::Mat, N> results;
    auto result = results.begin();
    for (auto const &pathname : pathnames)
        *result++ = pathname | load | pipeline;
    return results;
}

template<typename T>
std::vector<cv::Mat>
operator|(std::initializer_list<T> const &list,
          persistent_pipeline      const &pipeline)
{
    std::vector<cv::Mat> results;
    auto result = back_inserter(results);
    for (auto const &item : list)
    {
        if constexpr (std::is_same<T, std::filesystem::path>::value)
            *result++ = item | load | pipeline;
        else
            *result++ = item | pipeline;
    }
    return results;
}


}   // namespace opencv_pipeline

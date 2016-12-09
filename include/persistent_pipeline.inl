#pragma once

namespace opencv_pipeline {

typedef
enum { delay=true }
delay_result;

// create a persistent pipeline
inline
detail::persistent_pipeline operator|(delay_result, std::function<cv::Mat (cv::Mat const &)> rhs)
{
    return detail::persistent_pipeline(std::move(rhs));
}

inline
detail::persistent_pipeline operator|(delay_result, cv::Mat (*rhs)(cv::Mat const &))
{
    return detail::persistent_pipeline(rhs);
}

// run a persistent_pipeline
inline
cv::Mat operator|(cv::Mat lhs, detail::persistent_pipeline rhs)
{
    return rhs(std::move(lhs));
}

// construct pipes
inline
detail::persistent_pipeline operator|(detail::persistent_pipeline lhs, std::function<cv::Mat (cv::Mat const &)> rhs)
{
    return lhs.append(std::move(rhs));
}

inline
detail::persistent_pipeline operator|(std::function<cv::Mat (cv::Mat const &)> lhs, detail::persistent_pipeline rhs)
{
    return detail::persistent_pipeline(std::move(lhs));
}

inline
detail::persistent_pipeline operator|(detail::persistent_pipeline lhs, cv::Mat (*rhs)(cv::Mat const &))
{
    return lhs.append(rhs);
}

inline
detail::persistent_pipeline operator|(cv::Mat (*lhs)(cv::Mat const &), detail::persistent_pipeline rhs)
{
    return rhs.append(lhs);
}

}   // namespace opencv_pipeline

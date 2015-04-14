#pragma once

#include <stdexcept>

namespace cv_pipeline {

namespace exceptions {

class image_not_found : public std::runtime_error
{
  public:
    image_not_found(char const * const pathname)
      : runtime_error(pathname)
    {
    }
};

}   // namespace exceptions

}   // namespace cv_pipeline

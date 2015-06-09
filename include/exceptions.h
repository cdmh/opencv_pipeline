#pragma once

#include <stdexcept>

namespace opencv_pipeline {

namespace exceptions {

class file_not_found : public std::runtime_error
{
  public:
    file_not_found(char const * const pathname)
      : runtime_error(pathname)
    {
    }
};

class bad_image : public std::runtime_error
{
  public:
    bad_image()
      : runtime_error("bad_image")
    {
    }
};

class end_of_file : public std::exception
{
  public:
    end_of_file()
    {
    }
};

}   // namespace exceptions

}   // namespace opencv_pipeline

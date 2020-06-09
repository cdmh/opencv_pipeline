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

    file_not_found(std::filesystem::path pathname)
      : file_not_found(pathname.u8string().c_str())
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

class bad_video : public std::runtime_error
{
  public:
    bad_video(std::string error)
      : runtime_error(error)
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

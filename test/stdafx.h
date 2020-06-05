#pragma once

#include <iostream>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>


#if CV_MAJOR_VERSION==2
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#elif CV_MAJOR_VERSION==3
#include <opencv2/xfeatures2d/nonfree.hpp>
#endif

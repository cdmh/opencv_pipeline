OpenCV Pipeline
===============

A C++11 pipeline interface to OpenCV 2.4. See `develop` branch for current implementation.

Motivation
----------

Why write this:
```cpp
cv::Mat img = cv::imread("monalisa.jpg");
if (img.empty())
    throw std::runtime_error("file not found");

cvtColor(img, img, cv::COLOR_BGR2GRAY);
cvtColor(img, img, cv::COLOR_GRAY2BGR);
cv::flip(img, img, 1);
```

when you can write this:
```cpp
using namespace cv_pipeline;
cv::Mat img = "monalisa.jpg" | verify | gray | mirror;
```

Principles
==========

* Lightweight
* Safe
* Efficient

Lightweight
-----------

*OpenCV Pipeline* is designed to be lightweight and expression (somewhat functional). To change an image to gray scale, reflect it horizontally and save it back out, simply write:

```cpp
using namespace opencv_pipeline;
"colour.png" | verify | gray | mirror | save("result.png");
```

That's all there is to it. No fuss, no variables, just a simple pipeline of actions. If you want a copy of the result for further processing, assign the expression to a variable:

```cpp
using namespace opencv_pipeline;
auto gray_mirror = "colour.png" | verify | gray | mirror | save("result.png");
```

Safe
----

Inline verification is always available. Added `verify` at any point in the pipeline and if the previous step produced an invalid result (empty `Mat`), then an exception is thrown, with information where it is available.

It is mandatory to be explicit about error checking when loading an image within a pipeline. If you really want to avoid it, you can use `noverify`, but this is generally discouraged:

```cpp
auto image = "colour.png" | noverify | gray;
```

If the image load fails, and `noverify` is specified, then an empty image is passed to the next function in the pipeline. In this case, `gray` which calls OpenCV's `cvtColor` which will fail. If you are in exception free, then split the pipeline and use `noverify`:

```cpp
    auto image = "colour.png" | noverify;
    if (!image.empty())
        image = image | gray;
```

Efficient
---------

Some efficiency is compromised in the implementation with the hope that the compiler will be able to optimise the resulting code. OpenCV's reference counted `Mat` structures are a pain for optimisation, and return-by-value which should be a move operation isn't because of the ref-counted design.

Examples
========

Extracting Features from Keypoints
----------------------------------

Load a picture of the Mona Lisa, change it to gray scale, detect Harris Corner feature keypoints, extract SIFT feature descriptors and save the descriptors in a file result.png, ignoring save errors
```cpp
using namespace opencv_pipeline;
std::vector<cv::KeyPoint> keypoints;
"monalisa.jpg" | verify
    | gray
    | detect("HARRIS", keypoints) | extract("SIFT", keypoints)
    | save("result.png") | noverify;
```

Extracting  Features from Regions
---------------------------------

You want features from maximally stable regions instead of keypoints? Ok,
```cpp
using namespace opencv_pipeline;
std::vector<std::vector<cv::Point>> regions;
"monalisa.jpg" | verify
    | detect("MSCR", regions) | extract("SIFT", keypoints)
    | save("result.png") | noverify;
```


Reusing a pipeline
------------------
Reusing a pipeline is straightforward by storing the pipeline in a lambda function and call it for multiple images.

```cpp
auto pipeline = [](char const * const filename)->cv::Mat {
    std::vector<cv::KeyPoint> keypoints;
    return
    filename| verify
        | gray
        | detect("HARRIS", keypoints)
        | extract("SIFT", keypoints);
};

pipeline("monalisa.jpg")
    | save("monalisa-descriptors.jpg") | noverify;

pipeline("da_vinci_human11.jpg")
    | save("da_vinci_human11-descriptors.jpg") | noverify;
```

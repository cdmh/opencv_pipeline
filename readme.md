# OpenCV Pipeline

A C++11 pipeline interface to OpenCV 2.4. See `develop` branch for current implementation.

## Motivation

Why write this:
```cpp
auto img = cv::imread("monalisa.jpg");
if (img.empty())
    throw std::runtime_error("file not found");

cvtColor(img, img, cv::COLOR_BGR2GRAY);
cvtColor(img, img, cv::COLOR_GRAY2BGR);
cv::flip(img, img, 1);
```

when you can write this:
```cpp
using namespace cv_pipeline;
auto img = "monalisa.jpg" | verify | gray_bgr | mirror;
```

## Principles
* Lightweight
* Safe
* Efficient

## Lightweight
*OpenCV Pipeline* is designed to be lightweight and expressional (somewhat functional).
Display an image:
```cpp
using namespace opencv_pipeline;
"colour.png" | verify | show("image") | waitkey(5);
```

That's all there is to it. No fuss, no variables, just a simple pipeline of actions.
If you want a copy of the result for further processing, assign the expression to a variable:
```cpp
using namespace opencv_pipeline;
auto img = "colour.png" | verify | show("image") | waitkey(5);
```

## Safe
Inline verification is always available. Add `verify` at any point in the pipeline and if the previous step produced an invalid result (empty `Mat`), then an exception is thrown, with information where it is available.

It is mandatory to be explicit about error checking when loading an image within a pipeline.
If you really want to avoid processing errors, you can use `noverify`, but this is generally discouraged:

```cpp
auto image = "colour.png" | noverify | gray;
```

If the image load fails, and `noverify` is specified, then an empty image is passed to the next function in the pipeline. In this case, `gray` which calls OpenCV's `cvtColor` which will fail. If you are in an exception free environment, then split the pipeline and use `noverify`:

```cpp
    auto image = "colour.png" | noverify;
    if (!image.empty())
        image = image | gray;
```

## Efficient

Some efficiency is compromised in the implementation with the hope that the compiler will be able to optimise the resulting code. OpenCV's reference counted `Mat` structures are a pain for optimisation, and return-by-value which should be a move operation isn't because of the ref-counted design.

# Examples
---
### Extracting Features from Keypoints

Load a picture of the Mona Lisa, change it to gray scale, detect Harris Corner feature keypoints, extract SIFT feature descriptors and save the descriptors in a file result.png, ignoring save errors
```cpp
using namespace opencv_pipeline;
"monalisa.jpg" | verify
    | gray
    | keypoints("HARRIS") | descriptors("SIFT")
    | save("harris_sift.png") | noverify;
```

### Extracting  Features from Regions
You want features from maximally stable regions instead of keypoints? Ok,
```cpp
using namespace opencv_pipeline;
"monalisa.jpg" | verify
    | regions("MSCR") | descriptors("SIFT")
    | save("mscr_sift.png") | noverify;
```

### Reusing a pipeline
---
Use `delay` to create a pipeline object to store the function objects of the pipeline.
The pipeline can then be reused with different inputs.
```cpp
auto pipeline = delay | gray | mirror | show("Image") | waitkey(0);
"monalisa.jpg" | verify | pipeline;
"newton.jpg"   | verify | pipeline;
```

### Parameterised pipelines
---
Parameterising a pipeline is straightforward using a lambda function to store the pipeline,
and call it for multiple images.
```cpp
using namespace opencv_pipeline;
auto pipeline = [](char const * const filename)->cv::Mat {
    return
        filename| verify
            | gray
            | keypoints("HARRIS")
            | descriptors("SIFT");
};

pipeline("monalisa.jpg")
    | save("monalisa-harris-sift.jpg") | noverify;

pipeline("da_vinci_human11.jpg")
    | save("da_vinci_human11-harris-sift.png") | noverify;
```

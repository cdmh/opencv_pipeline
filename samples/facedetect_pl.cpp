#include "../include/opencv_pipeline.h"
#include "opencv2/objdetect.hpp"    // CascadeClassifier
#include <iostream>

namespace pipeline_samples {

using namespace std;
using namespace cv; //!!! this will go

static void help(const char** argv)
{
    cout << "\nThis program demonstrates the use of cv::CascadeClassifier class to detect objects (Face + eyes). You can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
        <<  argv[0]
        <<  "   [--cascade=<cascade_path> this is the primary trained classifier such as frontal face]\n"
            "   [--nested-cascade[=nested_cascade_path this an optional secondary classifier such as eyes]]\n"
            "   [--scale=<image scale greater or equal to 1, try 1.3 for example>]\n"
            "   [--try-flip]\n"
            "   [filename|camera_index]\n\n"
            "example:\n"
        <<  argv[0]
        <<  " --cascade=\"data/haarcascades/haarcascade_frontalface_alt.xml\" --nested-cascade=\"data/haarcascades/haarcascade_eye_tree_eyeglasses.xml\" --scale=1.3\n\n"
            "During execution:\n\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

cv::Mat detectAndDraw( Mat const & img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip );

string cascadeName;
string nestedCascadeName;

int main( int argc, const char** argv )
{
    using namespace opencv_pipeline;

    Mat frame, image;
    string inputName;
    bool tryflip;
    CascadeClassifier cascade, nestedCascade;
    double scale;

    cv::CommandLineParser parser(argc, argv,
        "{help h||}"
        "{cascade|data/haarcascades/haarcascade_frontalface_alt.xml|}"
        "{nested-cascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}"
        "{scale|1|}{try-flip||}{@filename||}"
    );
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
    cascadeName = parser.get<string>("cascade");
    nestedCascadeName = parser.get<string>("nested-cascade");
    scale = parser.get<double>("scale");
    if (scale < 1)
        scale = 1;
    tryflip = parser.has("try-flip");
    inputName = parser.get<string>("@filename");
    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }
    if (!nestedCascade.load(samples::findFileOrKeep(nestedCascadeName)))
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
    if (!cascade.load(samples::findFile(cascadeName)))
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        help(argv);
        return -1;
    }

    std::unique_ptr<video_pipeline> capture;

    if( inputName.empty() || (isdigit(inputName[0]) && inputName.size() == 1) )
    {
        int camera = inputName.empty() ? 0 : inputName[0] - '0';
        capture = make_unique<video_pipeline>(camera);
    }
    else if (!inputName.empty())
    {
        image = imread(samples::findFileOrKeep(inputName), IMREAD_COLOR);
        if (image.empty())
        {
            capture = make_unique<video_pipeline>(std::filesystem::path(samples::findFileOrKeep(inputName).c_str()));
            if (!capture->open())
                capture.reset();
        }
    }
    else
    {
        image = imread(samples::findFile("lena.jpg"), IMREAD_COLOR);
        if (image.empty())
        {
            cout << "Couldn't read lena.jpg" << endl;
            return 1;
        }
    }

    auto user_term = [](int c) {
        if( c == 27 || c == 'q' || c == 'Q' )
            throw exceptions::end_of_file();
    };

    auto detect_and_draw = std::bind( detectAndDraw, std::placeholders::_1, cascade, nestedCascade, scale, tryflip );

    if(capture)
    {
        *capture | load;
        cout << "Video capturing has been started ..." << endl;

        *capture
            | noop | verify // break on empty frame
            | clone
            | detect_and_draw
            | show("result")
            | waitkey(10, user_term)
            | play;
    }
    else
    {
        cout << "Detecting face(s) in " << inputName << endl;
        if( !image.empty() )
        {
            detectAndDraw( image, cascade, nestedCascade, scale, tryflip );
            waitKey(0);
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            FILE* f = fopen( inputName.c_str(), "rt" );
            if( f )
            {
                char buf[1000+1];
                while( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf);
                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';
                    cout << "file " << buf << endl;

                    try
                    {
                        std::filesystem::path(buf) | load | verify
                            | detect_and_draw
                            | show("result")
                            | waitkey(0, user_term);
                    }
                    catch(exceptions::end_of_file &)
                    {
                        break;
                    }
                    catch(exceptions::bad_image &)
                    {
                        cerr << "Aw snap, couldn't read image " << buf << endl;
                    }
                }
                fclose(f);
            }
        }
    }

    return 0;
}

cv::Mat detectAndDraw( Mat const & img, CascadeClassifier& cascade,
                    CascadeClassifier& nestedCascade,
                    double scale, bool tryflip )
{
    using namespace opencv_pipeline;

    double t = 0;
    vector<Rect> faces, faces2;
    const static Scalar colors[] =
    {
        Scalar(255,0,0),
        Scalar(255,128,0),
        Scalar(255,255,0),
        Scalar(0,255,0),
        Scalar(0,128,255),
        Scalar(0,255,255),
        Scalar(0,0,255),
        Scalar(255,0,255)
    };

    double fx = 1 / scale;
    auto smallImg = img
        | gray
        | resize(fx, fx, INTER_LINEAR_EXACT)
        | equalizeHist;

    auto detect_multi_scale = std::bind(
        [&cascade](cv::Mat image, std::vector<Rect>& objects) -> cv::Mat {
            cascade.detectMultiScale(
                image, objects,
                1.1, 2, CASCADE_SCALE_IMAGE, Size(30, 30));
            return image;
        }, std::placeholders::_1, std::placeholders::_2);

    t = (double)getTickCount();
    smallImg = smallImg
        | std::bind(detect_multi_scale, std::placeholders::_1, std::ref(faces));

    if( tryflip )
    {
        smallImg = smallImg
            | mirror
            | std::bind(detect_multi_scale, std::placeholders::_1, std::ref(faces2));
        for( vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end(); ++r )
        {
            faces.push_back(Rect(smallImg.cols - r->x - r->width, r->y, r->width, r->height));
        }
    }
    t = (double)getTickCount() - t;
    printf( "detection time = %g ms\n", t*1000/getTickFrequency());
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Rect r = faces[i];
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i%8];
        int radius;

        double aspect_ratio = (double)r.width/r.height;
        if( 0.75 < aspect_ratio && aspect_ratio < 1.3 )
        {
            center.x = cvRound((r.x + r.width*0.5)*scale);
            center.y = cvRound((r.y + r.height*0.5)*scale);
            radius = cvRound((r.width + r.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
        else
            rectangle( img, Point(cvRound(r.x*scale), cvRound(r.y*scale)),
                       Point(cvRound((r.x + r.width-1)*scale), cvRound((r.y + r.height-1)*scale)),
                       color, 3, 8, 0);
        if( nestedCascade.empty() )
            continue;
        smallImgROI = smallImg( r );
        smallImgROI
            | std::bind(detect_multi_scale, std::placeholders::_1, std::ref(nestedObjects));

        detect_multi_scale(smallImgROI, nestedObjects);

        for ( size_t j = 0; j < nestedObjects.size(); j++ )
        {
            Rect nr = nestedObjects[j];
            center.x = cvRound((r.x + nr.x + nr.width*0.5)*scale);
            center.y = cvRound((r.y + nr.y + nr.height*0.5)*scale);
            radius = cvRound((nr.width + nr.height)*0.25*scale);
            circle( img, center, radius, color, 3, 8, 0 );
        }
    }
    return img;
}

}   // namespace pipeline_samples

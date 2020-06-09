namespace opencv_samples {
int main( int argc, const char** argv );
}   // namespace opencv_samples

namespace pipeline_samples {
int main( int argc, const char** argv );
}   // namespace pipeline_samples

int main(int argc, const char **argv)
{
    return pipeline_samples::main(argc, argv);
//    return opencv_samples::main(argc, argv);
}


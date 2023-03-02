#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    // If no image name given on executable launch, give usage instructions
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    // Try reading image from argv[1]
    Mat image;
    image = imread( argv[1], 1 );

    // If no data, give error message
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    // Create window and show image
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>


namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;

using namespace std;
using namespace cv;
 
void undistortCore( Mat& img_u, Mat& img_d, Mat& H, 
                    size_t xPartitionStart, size_t xPartitionEnd){

    double N = img_u.rows;
    for (double m=xPartitionStart; m<xPartitionEnd; m++){ // double for compatibility when
        for (double n=0; n<N; n++){                       // in matmul later on
            // build undistorted, homogeneous coordinate vector
            Mat xu = (Mat_<double>(3, 1) << m, n, 1);

            // coordinate transform
            Mat xd = H*xu;
            
            // convert back to inhom coords
            xd = xd / xd.at<double>(2,0);

            // round to integer indexes
            xd.convertTo(xd, CV_32S);

            // use transformed coords to get pixel value from distorted image
            int x = xd.at<int>(0,0); 
            int y = xd.at<int>(1,0); 
            img_u.at<Vec3b>(m, n) = img_d.at<Vec3b>(Point(y, x)); // Point uses reverse order!
        }
    }
}

Mat pointwiseUndistort( 
    Mat& img_d, 
    Mat& H, 
    int M,
    int N
){
    Mat img_u(M, N, CV_8UC3); // prepare return image
    
    size_t partitionSize = M/4;

    std::thread th1(undistortCore, ref(img_u), ref(img_d), ref(H), 0*partitionSize, 1*partitionSize);
    std::thread th2(undistortCore, ref(img_u), ref(img_d), ref(H), 1*partitionSize, 2*partitionSize);
    std::thread th3(undistortCore, ref(img_u), ref(img_d), ref(H), 2*partitionSize, 3*partitionSize);
    std::thread th4(undistortCore, ref(img_u), ref(img_d), ref(H), 3*partitionSize, M);
    
    th1.join();
    th2.join();
    th3.join();
    th4.join();

    return img_u;
}       

PYBIND11_MODULE(cppmodule, m){
        m.doc() = "Docstring for pointwiseUndistort function";
        m.def("pointwiseUndistort", [](
            py::array_t<imgScalar>& pyImg_d, 
            py::array_t<matScalar>& pyH, 
            py::tuple img_u_shape
            )
            {
                // -- Data type management
                // link pyImg_d data to cv::Mat object img
                Mat img_d(
                    pyImg_d.shape(0),               // rows
                    pyImg_d.shape(1),               // cols
                    CV_8UC3,                        // data type
                    (imgScalar*)pyImg_d.data());    // data pointer

                // link H data to cv::Mat object
                Mat H(
                    pyH.shape(0),                   // rows
                    pyH.shape(1),                   // cols
                    CV_64FC1,                       // data type
                    (matScalar*)pyH.data());        // data pointer

                int M = img_u_shape[0].cast<int>();
                int N = img_u_shape[1].cast<int>();

                // -- Call C++ function with C-types
                return pointwiseUndistort(img_d, H, M, N);
            }
        );

        // Binding cv::Mat class and defining a type conversion to buffer object for returning images to python
        py::class_<cv::Mat>(m, "Mat", py::buffer_protocol()) 
            .def_buffer([](cv::Mat &im) -> py::buffer_info {
                    return py::buffer_info(
                        im.data,                                            // pointer to data
                        sizeof(unsigned char),                              // item size
                        py::format_descriptor<unsigned char>::format(),     // item descriptor
                        3,                                                  // matrix dimensionality
                        {                                                   // buffer dimensions
                            im.rows, 
                            im.cols, 
                            im.channels()
                        },          
                        {                                                    // strides in bytes
                            sizeof(unsigned char) * im.channels() * im.cols, 
                            sizeof(unsigned char) * im.channels(),
                            sizeof(unsigned char)
                        }
                    );
                })
            ;
    }

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;

namespace cv {
    Mat pointwiseUndistort( py::array_t<imgScalar>& pyImg_d, 
                            py::array_t<matScalar>& pyH, 
                            py::tuple img_u_shape ){

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
    
        // -- Algorithm
        Mat img_u(M, N, CV_8UC3); // prepare return image

        for (int m=0; m<M; m++){ // m = {0, 1, 2, ... , M-1}
            for (int n=0; n<N; n++){ 
                Mat xu = (Mat_<int>(3, 1) << m, n, 1);
                Mat xu_double;
                xu.convertTo(xu_double, CV_64F);

                // coordinate transform
                Mat xd_double = H*xu_double;
                
                // convert back to inhom coords
                xd_double = xd_double / xd_double.at<double>(2,0);

                // round to integer indexes
                Mat xd_int;
                xd_double.convertTo(xd_int, CV_32S);

                // use transformed coords to get pixel value from distorted image
                int x = xd_int.at<int>(0,0); 
                int y = xd_int.at<int>(1,0); 

                Vec3b rgb_vect = img_d.at<Vec3b>(Point(y, x)); //reverso!
                img_u.at<Vec3b>(m, n) = rgb_vect;
            }
        }
        return img_u;
    }       

    PYBIND11_MODULE(cppmodule, m){
        m.def("pointwiseUndistort", &pointwiseUndistort, py::return_value_policy::automatic);
        m.doc() = "Docstring for pointwiseUndistort function";

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
                            sizeof(unsigned char) * im.channels() * im.cols, // (issue with padding)
                            sizeof(unsigned char) * im.channels(),
                            sizeof(unsigned char)
                        }
                    );
                })
            ;
    }
}
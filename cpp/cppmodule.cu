#include <stdio.h>
#include <opencv2/opencv.hpp>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
#include <chrono>

// namespace py = pybind11;

typedef std::uint8_t imgScalar;
typedef double matScalar;

using namespace std;
using namespace cv;
 
 /*
__global__ void undistortKernel( Mat& img_u, Mat& img_d, Mat& H){
}*/

__global__ void add(int* x, int* y, int* r)
{
    int i = threadIdx.x;
    r[i] = x[i] + y[i];
}

/*
Mat pointwiseUndistort( py::array_t<imgScalar>& pyImg_d, 
                        py::array_t<matScalar>& pyH, 
                        py::tuple img_u_shape ){
*/

int main(){
    // -- Data type management
    // link pyImg_d data to cv::Mat object img
    /*
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
    */
    
    // --- unified memory test ---
    int N = 1<<20;

    int arrX[] = {1,2,3,4};
    int arrY[] = {1,2,3,4};
    int arrR[4];

    int* x;
    int* y;
    int* r;

    x = arrX;
    y = arrY;
    r = arrR;

    auto start = chrono::steady_clock::now();

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, sizeof(arrX));
    cudaMallocManaged(&y, sizeof(arrY));
    cudaMallocManaged(&r, sizeof(arrR));

    auto end = chrono::steady_clock::now();

    // Run kernel on 1M elements on the GPU
    add<<<1, 4>>>(arrX, arrY, arrR);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    

    // timing output
    cout << "Elapsed time in microseconds: "
    << chrono::duration_cast<chrono::microseconds>(end - start).count()
    << " µs" << endl;

    // Free memory
    cudaFree(arrX);
    cudaFree(arrY);
    cudaFree(arrR);


    return 0;//img_d;//img_u;
}       
/*
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
*/
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

__global__ void vectorAdd(int* a, int* b, int* c) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
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
    
    // --- vectorAdd test ---
    int a[] = {1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12};
	int b[] = {1,2,3,4,5,6,7,8,9,10,11,12,1,2,3,4,5,6,7,8,9,10,11,12};
	auto NUMBER_OF_VECTORS = sizeof(a) / sizeof(int);
	int c[NUMBER_OF_VECTORS] = {0};

	// create pointers into the GPU
	int* cudaA;
	int* cudaB;
	int* cudaC;

	// allocate memory in the GPU
	cudaMallocManaged(&cudaA, sizeof(a));
	cudaMallocManaged(&cudaB, sizeof(b));
	cudaMallocManaged(&cudaC, sizeof(c));
    
    auto start = chrono::steady_clock::now();

	// copy into GPU
	cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, sizeof(a), cudaMemcpyHostToDevice);

	auto GRID_SIZE = 1; 				 	// number of blocks in grid
	auto BLOCK_SIZE = NUMBER_OF_VECTORS; 	// size of elements in block

    // CPU waits till kernel finished executing before moving on to next line of host code
	vectorAdd <<< GRID_SIZE, BLOCK_SIZE >>> (cudaA, cudaB, cudaC);
	
    // copy back out of GPU
	cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);



    auto end = chrono::steady_clock::now(); 
    cout << "Elapsed time in microseconds: "
        << chrono::duration_cast<chrono::microseconds>(end - start).count()
        << " µs" << endl;

    // print computation result
	for (int i = 0; i < NUMBER_OF_VECTORS; i++) {
		std::cout << c[i] << " ";
    }
    std::cout << std::endl;



    // free memory
    cudaFree(cudaA);
    cudaFree(cudaB);
    cudaFree(cudaC);

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
#include <iostream>
#include <cmath>
#include <iomanip>

#include "KFMFastFourierTransform.hh"
#include "KFMMultidimensionalFastFourierTransform.hh"
#include "KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh"

#include "KEMCout.hh"
#include "KFMMessaging.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int ndim = 4;
    const unsigned int p = 12;
    const unsigned int div = 8;
    const unsigned int zmask = 2;
    const unsigned int p2_size = (p+1)*(p+1);
    const unsigned int spatial = 2*div*(zmask + 1);
    const unsigned int dim_size[ndim] = {p2_size,spatial,spatial,spatial};
    const unsigned int total_size_gpu = dim_size[0]*dim_size[1]*dim_size[2]*dim_size[3];
    const unsigned int total_size_cpu = dim_size[0]*dim_size[1]*dim_size[2]*dim_size[3];

    std::cout<<"p-size = "<<p2_size<<std::endl;
    std::cout<<"spatial size = "<<spatial<<std::endl;
    std::cout<<"number of 1-d transforms = "<<p2_size*spatial*spatial*3<<std::endl;

    std::complex<double>* raw_data_gpu = new std::complex<double>[total_size_gpu];
    std::complex<double>* raw_data_cpu = new std::complex<double>[total_size_cpu];

    KFMArrayWrapper< std::complex<double>, ndim> input_gpu(raw_data_gpu, dim_size);
    KFMArrayWrapper< std::complex<double>, ndim-1> input_cpu(raw_data_cpu, &(dim_size[1]) );

    //fill up the array with a signal
    int index[ndim];
    int count = 0;
    index[0] = 0;

    for(unsigned int i=0; i<total_size_cpu; i++)
    {
        raw_data_cpu[i] = count % 13;
        count++;
    }

    for(unsigned int i=0; i<total_size_gpu; i++)
    {
        raw_data_gpu[i] = count % 13;
        count++;
    }

    kfmout<<"--------------------------------------------------------------"<<kfmendl;

    KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>* fft_gpu = new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>();

//    fft_gpu->SetWriteOutHostDataFalse();
//    fft_gpu->SetReadOutDataToHostFalse();

    fft_gpu->SetForward();
    fft_gpu->SetInput(&input_gpu);
    fft_gpu->SetOutput(&input_gpu);
    fft_gpu->Initialize();

    clock_t start, end;
    //double duration;
    start = clock();

    fft_gpu->ExecuteOperation();

    end = clock();
    double time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds

    std::cout<<"time for fft on gpu = "<<time<<std::endl;


    KFMMultidimensionalFastFourierTransform<3>* fft_cpu = new KFMMultidimensionalFastFourierTransform<3>();

    fft_cpu->SetForward();
    fft_cpu->SetInput(&input_cpu);
    fft_cpu->SetOutput(&input_cpu);
    fft_cpu->Initialize();

    start = clock();
    for(unsigned int i=0; i<p2_size; i++)
    {
        fft_cpu->ExecuteOperation();
    }
    end = clock();
    time = ((double)(end - start))/CLOCKS_PER_SEC; // time in seconds
    std::cout<<"time for fft on cpu = "<<time<<std::endl;

    delete fft_gpu;
    delete fft_cpu;

    return 0;
}

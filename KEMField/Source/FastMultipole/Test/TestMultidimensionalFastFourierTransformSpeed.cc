#include <getopt.h>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#ifdef KEMFIELD_USE_REALTIME_CLOCK
#include <time.h>
#endif

#include <cmath>

#include "KFMFastFourierTransform.hh"
#include "KFMMultidimensionalFastFourierTransform.hh"
#include "KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh"

#include "KEMCout.hh"
#include "KFMMessaging.hh"

using namespace KEMField;


#ifdef KEMFIELD_USE_REALTIME_CLOCK
timespec diff(timespec start, timespec end)
{
    timespec temp;
    if( (end.tv_nsec-start.tv_nsec) < 0)
    {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    }
    else
    {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}
#endif

int main(int argc, char** argv)
{

#ifdef KEMFIELD_USE_REALTIME_CLOCK

  std::string usage =
    "\n"
    "Usage: TestMultidimensionalFastFourierTransformSpeed <options>\n"
    "\n"
    "This program computes a batch of 3D fast fourier transforms using the OpenCL device. \n"
    "\tAvailable options:\n"
    "\t -h, --help               (shows this message and exits)\n"
    "\t -n, --n-samples          (number of times to run FFTs)\n"
    "\t -p, --degree             (p, size of 4th dimension is (p+1)(p+2)/2 )\n"
    "\t -d, --divisions          (d, size of spatial dimensions are 2*d*(z+1) )\n"
    "\t -z  --zeromask           (z, size of spatial dimensions are 2*d*(z+1) )\n"
    "\t -c  --cpu                (use cpu instead of OpenCL device)  \n"
    ;

    const unsigned int ndim = 4;
    unsigned int n_samples = 10;
    unsigned int p = 16;
    unsigned int div = 4;
    unsigned int zmask = 2;
    unsigned int use_cpu = 0;


    static struct option longOptions[] =
    {
        {"help", no_argument, 0, 'h'},
        {"n-samples", required_argument, 0, 'n'},
        {"degree", required_argument, 0, 'p'},
        {"divisions", required_argument, 0, 'd'},
        {"zeromask", required_argument, 0, 'z'},
        {"cpu", required_argument, 0, 'c'},
    };

    static const char *optString = "hn:p:d:z:c";

    while(1)
    {
        char optId = getopt_long(argc, argv,optString, longOptions, NULL);
        if(optId == -1) break;
        switch(optId)
        {
            case('h'): // help
            std::cout<<usage<<std::endl;
            return 0;
            case('n'):
            n_samples = atoi(optarg);
            break;
            case('p'):
            p = atoi(optarg);
            break;
            case('d'):
            div = atoi(optarg);
            break;
            case('z'):
            zmask = atoi(optarg);
            break;
            case('c'):
            use_cpu = 1;
            break;
            default:
            std::cout<<usage<<std::endl;
            return 1;
        }
    }

    const unsigned int p2_size = (p+1)*(p+2)/2;
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
    int count = 0;
    for(unsigned int i=0; i<total_size_cpu; i++)
    {
        raw_data_cpu[i] = count % 13;
        count++;
    }

    count = 0;
    for(unsigned int i=0; i<total_size_gpu; i++)
    {
        raw_data_gpu[i] = count % 13;
        count++;
    }

    if(use_cpu == 1)
    {

        KFMMultidimensionalFastFourierTransform<3>* fft_cpu = new KFMMultidimensionalFastFourierTransform<3>();

        fft_cpu->SetForward();
        fft_cpu->SetInput(&input_cpu);
        fft_cpu->SetOutput(&input_cpu);
        fft_cpu->Initialize();

        timespec start, end;
        clock_t cstart, cend;
        double time, ctime;
        clock_gettime(CLOCK_REALTIME, &start);
        cstart = clock();

        for(unsigned int s=0; s<n_samples; s++)
        {
            for(unsigned int i=0; i<p2_size; i++)
            {
                fft_cpu->ExecuteOperation();
            }
        }


        clock_gettime(CLOCK_REALTIME, &end);
        cend = clock();
        timespec temp = diff(start, end);
        time = (double)temp.tv_sec + (double)(temp.tv_nsec*1e-9);
        ctime = ((double)(cend - cstart))/CLOCKS_PER_SEC; // time in seconds

        double tp_sample = time/(double)n_samples;
        double tp_sample2 = ctime/(double)n_samples;

        std::cout << "CPU: Real time required per FFT sample (sec): " << tp_sample << std::endl;
        std::cout << "CPU: Process/CPU time required per FFT sample (sec): " << tp_sample2 << std::endl;

        delete fft_cpu;

    }
    else
    {

        KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>* fft_gpu = new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>();

        fft_gpu->SetForward();
        fft_gpu->SetInput(&input_gpu);
        fft_gpu->SetOutput(&input_gpu);
        fft_gpu->Initialize();

        timespec start, end;
        clock_t cstart, cend;
        double time, ctime;
        clock_gettime(CLOCK_REALTIME, &start);
        cstart = clock();

        for(unsigned int s=0; s<n_samples; s++)
        {
            fft_gpu->ExecuteOperation();
        }

        clock_gettime(CLOCK_REALTIME, &end);
        cend = clock();
        timespec temp = diff(start, end);
        time = (double)temp.tv_sec + (double)(temp.tv_nsec*1e-9);
        ctime = ((double)(cend - cstart))/CLOCKS_PER_SEC; // time in seconds

        double tp_sample = time/(double)n_samples;
        double tp_sample2 = ctime/(double)n_samples;

        std::cout << "GPU: Real time required per FFT sample (sec): " << tp_sample << std::endl;
        std::cout << "GPU: Process/CPU time required per FFT sample (sec): " << tp_sample2 << std::endl;

        delete fft_gpu;
    }

    #else
        (void) argc;
        (void) argv;
        std::cout<<"To run this test please compile KEMField with the KEMField_USE_REALTIME_CLOCK flag enabled"<<std::endl;
    #endif

    return 0;
}

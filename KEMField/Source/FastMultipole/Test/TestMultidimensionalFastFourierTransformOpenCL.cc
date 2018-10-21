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
    const unsigned int p = 8;
    const unsigned int stride = ((p+1)*(p+2))/2;
    const unsigned int d = 3;
    const unsigned int z = 1;
    const unsigned int div_size = 2*d*(z+1);


    const unsigned int batch_size = stride;
    const unsigned int ndim = 4;
    const unsigned int dim_size[ndim] = {batch_size,div_size,div_size,div_size};

    const unsigned int total_size = dim_size[0]*dim_size[1]*dim_size[2]*dim_size[3];

    double spatial_size = dim_size[1]*dim_size[2]*dim_size[3];

    std::complex<double>* raw_data = new std::complex<double>[total_size];
    KFMArrayWrapper< std::complex<double>, ndim> input(raw_data, dim_size);

    //fill up the array with a signal

    for(unsigned int i=0; i<total_size; i++)
    {
        raw_data[i] = i;
    }

    int index[ndim];
    int count = 0;

    for(unsigned int a=0; a<batch_size; a++)
    {
        index[0] = a;
        count = 0;
        //kfmout<<"original data = "<<kfmendl;
        for(unsigned int i=0; i<dim_size[1]; i++)
        {
            index[1] = i;
            for(unsigned int j=0; j<dim_size[2]; j++)
            {
                index[2] = j;

                for(unsigned int k=0; k<dim_size[3]; k++)
                {
                    index[3] = k;
                    input[index] = std::complex<double>(count%13,count%3);
                    //kfmout<<input[index]<<", ";
                    count++;
                }
                //kfmout<<kfmendl;
            }
            //kfmout<<kfmendl;

        }
    }

    //kfmout<<"--------------------------------------------------------------"<<kfmendl;

    KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>* fft_eng = new KFMBatchedMultidimensionalFastFourierTransform_OpenCL<3>();

    fft_eng->SetForward();
    fft_eng->SetInput(&input);
    fft_eng->SetOutput(&input);

    fft_eng->Initialize();

    fft_eng->ExecuteOperation();

    kfmout<<"DFT of data = "<<kfmendl;
//    index[0] = 0;
//    for(unsigned int i=0; i<dim_size[1]; i++)
//    {
//        index[1] = i;
//        for(unsigned int j=0; j<dim_size[2]; j++)
//        {
//            index[2] = j;

//            for(unsigned int k=0; k<dim_size[3]; k++)
//            {
//                index[3] = k;
//               kfmout<<input[index]<<", ";
//            }
//            kfmout<<kfmendl;
//        }
//        kfmout<<kfmendl;
//    }

//    kfmout<<"--------------------------------------------------------------"<<kfmendl;

    fft_eng->SetBackward();
    fft_eng->ExecuteOperation();

    //kfmout<<"IDFT of DFT of data = "<<kfmendl;
    count =0;
    double l2_norm = 0;
    double norm = spatial_size;

    for(unsigned int a=0; a<batch_size; a++)
    {
        l2_norm = 0;
        count = 0;
        index[0] = a;
        for(unsigned int i=0; i<dim_size[1]; i++)
        {
            index[1] = i;
            for(unsigned int j=0; j<dim_size[2]; j++)
            {
                index[2] = j;

                for(unsigned int k=0; k<dim_size[3]; k++)
                {
                    index[3] = k;
                    //kfmout<<input[index]/norm<<", ";

                    std::complex<double> del = input[index]/norm;
                    del -= std::complex<double>(count%13,count%3);

                    l2_norm += std::real(del)*std::real(del) + std::imag(del)*std::imag(del);

                    count++;

                }
                //kfmout<<kfmendl;
            }
            //kfmout<<kfmendl;
        }

        std::cout<<"L2 norm difference = "<<std::sqrt(l2_norm)<<std::endl;

    }


    delete fft_eng;

    return 0;
}

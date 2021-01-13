#include "KFMFastFourierTransform.hh"
#include "KFMMultidimensionalFastFourierTransform.hh"

#include <cmath>
#include <iomanip>
#include <iostream>

#ifdef KEMFIELD_USE_FFTW
#include "KFMMultidimensionalFastFourierTransformFFTW.hh"
#define FFT_TYPE KFMMultidimensionalFastFourierTransformFFTW<3>
#else
#define FFT_TYPE KFMMultidimensionalFastFourierTransform<3>
#endif

#include "KEMCout.hh"
#include "KFMMessaging.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int ndim = 3;
    const unsigned int dim_size[ndim] = {18, 18, 18};
    const unsigned int total_size = dim_size[0] * dim_size[1] * dim_size[2];

    auto* raw_data = new std::complex<double>[total_size];
    KFMArrayWrapper<std::complex<double>, ndim> input(raw_data, dim_size);

    //fill up the array with a signal

    for (unsigned int i = 0; i < total_size; i++) {
        raw_data[i] = i;
    }

    int index[ndim];
    int count = 0;
    kfmout << "original data = " << kfmendl;
    for (unsigned int i = 0; i < dim_size[0]; i++) {
        index[0] = i;
        for (unsigned int j = 0; j < dim_size[1]; j++) {
            index[1] = j;

            for (unsigned int k = 0; k < dim_size[2]; k++) {
                index[2] = k;
                input[index] = std::complex<double>(count % 13, count % 17);
                kfmout << input[index] << ", ";
                count++;
            }
            kfmout << kfmendl;
        }
        kfmout << kfmendl;
    }

    kfmout << "--------------------------------------------------------------" << kfmendl;

    auto* fft2d = new FFT_TYPE();

    fft2d->SetForward();
    fft2d->SetInput(&input);
    fft2d->SetOutput(&input);

    fft2d->Initialize();

    fft2d->ExecuteOperation();

    kfmout << "DFT of data = " << kfmendl;
    for (unsigned int i = 0; i < dim_size[0]; i++) {
        index[0] = i;
        for (unsigned int j = 0; j < dim_size[1]; j++) {
            index[1] = j;

            for (unsigned int k = 0; k < dim_size[2]; k++) {
                index[2] = k;
                kfmout << input[index] << ", ";
            }
            kfmout << kfmendl;
        }
        kfmout << kfmendl;
    }

    kfmout << "--------------------------------------------------------------" << kfmendl;

    fft2d->SetBackward();
    fft2d->SetInput(&input);
    fft2d->SetOutput(&input);
    fft2d->Initialize();

    fft2d->ExecuteOperation();

    kfmout << "IDFT of DFT of data = " << kfmendl;
    double norm = total_size;
    count = 0;
    double l2_norm = 0;
    for (unsigned int i = 0; i < dim_size[0]; i++) {
        index[0] = i;
        for (unsigned int j = 0; j < dim_size[1]; j++) {
            index[1] = j;

            for (unsigned int k = 0; k < dim_size[2]; k++) {
                index[2] = k;
                //kfmout<<input[index]/norm<<", ";

                std::complex<double> del = input[index] / norm;
                del -= std::complex<double>(count % 13, count % 17);
                l2_norm += std::real(del) * std::real(del) + std::imag(del) * std::imag(del);
                count++;
            }
            //kfmout<<kfmendl;
        }
        //kfmout<<kfmendl;
    }


    std::cout << "L2 norm difference = " << std::sqrt(l2_norm) << std::endl;

    //    KFMFastFourierTransform* fft = new KFMFastFourierTransform();
    //    fft->SetSize(N);
    //    fft->SetForward();
    //    fft->SetInput(&input);
    //    fft->SetOutput(&input);

    //    fft->Initialize();
    //    fft->ExecuteOperation();

    //    kfmout<<"DFT of data = "<<kfmendl;
    //    for(unsigned int i=0; i<N; i++)
    //    {
    //        kfmout<<"data["<<i<<"] = "<<raw_data[i]<<kfmendl;
    //    }

    //    fft->SetBackward();
    //    fft->Initialize();
    //    fft->ExecuteOperation();

    //    //the fft does not take care of the normalization, so we do that here
    //    double norm = 1.0/((double)N);
    //    kfmout<<"IDFT of DFT of data = "<<kfmendl;
    //    for(unsigned int i=0; i<N; i++)
    //    {
    //        kfmout<<"data["<<i<<"] = "<<norm*raw_data[i]<<kfmendl;
    //    }

    delete fft2d;

    return 0;
}

#include <iostream>
#include <cmath>
#include <iomanip>

#include "KFMFastFourierTransform.hh"

#include "KFMMessaging.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int N = 14;

    unsigned int dim[1];
    dim[0] = N;
    std::complex<double>* raw_data = new std::complex<double>[N];
    KFMArrayWrapper< std::complex<double>, 1> input(raw_data, dim);

    //fill up the array with a signal
    std::cout<<"original data = "<<std::endl;
    for(unsigned int i=0; i<N; i++)
    {
        raw_data[i] = std::complex<double>(i%3, i%5);
        std::cout<<"data["<<i<<"] = "<<raw_data[i]<<std::endl;
    }


    KFMFastFourierTransform* fft = new KFMFastFourierTransform();
    fft->SetSize(N);
    fft->SetForward();
    fft->SetInput(&input);
    fft->SetOutput(&input);

    fft->Initialize();
    fft->ExecuteOperation();

    std::cout<<"DFT of data = "<<std::endl;
    for(unsigned int i=0; i<N; i++)
    {
        std::cout<<"data["<<i<<"] = "<<raw_data[i]<<std::endl;
    }

    fft->SetBackward();
    fft->Initialize();
    fft->ExecuteOperation();

//    //the fft does not take care of the normalization, so we do that here
//    double norm = 1.0/((double)N);
//    std::cout<<"IDFT of DFT of data = "<<std::endl;
//    for(unsigned int i=0; i<N; i++)
//    {
//        std::cout<<"data["<<i<<"] = "<<norm*raw_data[i]<<std::endl;
//    }


    std::cout<<"-----------------------------------------"<<std::endl;


    //fill up the array with a signal
    std::cout<<"original data = "<<std::endl;
    for(unsigned int i=0; i<N; i++)
    {
        raw_data[i] = std::complex<double>(i%3, -1.0*(i%5) );
        std::cout<<"data["<<i<<"] = "<<raw_data[i]<<std::endl;
    }

    fft->SetForward();
    fft->Initialize();
    fft->ExecuteOperation();

    std::cout<<"DFT of data = "<<std::endl;
    for(unsigned int i=0; i<N; i++)
    {
        std::cout<<"data["<<i<<"] = "<<raw_data[i]<<std::endl;
    }


    return 0;
}

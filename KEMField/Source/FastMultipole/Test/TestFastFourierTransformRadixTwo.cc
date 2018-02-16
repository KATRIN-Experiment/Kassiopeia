#include <iostream>
#include <cmath>
#include <iomanip>

#include "KFMBitReversalPermutation.hh"
#include "KFMFastFourierTransformUtilities.hh"
#include "KFMMessaging.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int N = 8;

    kfmout<<"next biggest power of two after 187 is "<<KFMBitReversalPermutation::NextLowestPowerOfTwo(187)<<kfmendl;
    kfmout<<"next biggest power of two after 2345 is "<<KFMBitReversalPermutation::NextLowestPowerOfTwo(2345)<<kfmendl;
    kfmout<<"next biggest power of two after 78456 is "<<KFMBitReversalPermutation::NextLowestPowerOfTwo(78456)<<kfmendl;

    unsigned int index_arr[N];
    std::complex<double> arr[N];
    std::complex<double> arr_orig[N];
    std::complex<double> twiddle[N];

//    KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(N,index_arr);

//    //fill up the array with a signal
//    kfmout<<"Original array = "<<kfmendl;
//    for(unsigned int i=0; i<N; i++)
//    {
//        arr[i] = std::complex<double>( i , 0);
//        arr_orig[i] = arr[i];
//        kfmout<<arr[i]<<kfmendl;
//    }

//    //compute twiddle factors
//    KFMFastFourierTransformUtilities::ComputeTwiddleFactors(N,twiddle);

//    kfmout<<"twiddle factors = "<<kfmendl;
//    for(unsigned int i=0; i<N/2; i++)
//    {
//        kfmout<<"t("<<i<<") = "<<twiddle[i]<<kfmendl;
//    }

//    //perform bit reversed address permutation
//    KFMBitReversalPermutation::PermuteArray< std::complex<double> >(N, index_arr, arr);

//    //do the radix-2 FFT
//    KFMFastFourierTransformUtilities::FFTRadixTwo(N, (double*)&(arr[0]), (double*) &(twiddle[0]) );

//    kfmout<<"DFT'd array = "<<kfmendl;
//    for(unsigned int i=0; i<N; i++)
//    {
//        kfmout<<arr[i]<<kfmendl;
//    }

//    //now we'll do the inverse transform

//    //conjugate the twiddle factors
//    for(unsigned int i=0; i<N; i++)
//    {
//        twiddle[i] = std::conj(twiddle[i]);
//    }

//    //perform bit reversed address permutation
//    KFMBitReversalPermutation::PermuteArray< std::complex<double> >(N, index_arr, arr);

//    //do the radix-2 FFT
//    KFMFastFourierTransformUtilities::FFTRadixTwo(N, (double*)&(arr[0]), (double*) &(twiddle[0]) );

//    //normalize
//    for(unsigned int i=0; i<N; i++)
//    {
//        arr[i] *= 1.0/((double)N);
//    }

//    kfmout<<"difference between original and IDFT of the DFT'd array = "<<kfmendl;
//    for(unsigned int i=0; i<N; i++)
//    {
//        kfmout<< arr_orig[i] - arr[i]<<kfmendl;
//    }

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


    std::cout<<"---------------------------------------------------------------"<<std::endl;
    std::cout<<"testing DIT followed by DIF"<<std::endl;


    KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(N,index_arr);

    //fill up the array with a signal
    kfmout<<"Original array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        arr[i] = std::complex<double>( i , 0);
        arr_orig[i] = arr[i];
        kfmout<<arr[i]<<kfmendl;
    }

    //compute twiddle factors
    KFMFastFourierTransformUtilities::ComputeTwiddleFactors(N,twiddle);

//    kfmout<<"twiddle factors = "<<kfmendl;
//    for(unsigned int i=0; i<N/2; i++)
//    {
//        kfmout<<"t("<<i<<") = "<<twiddle[i]<<kfmendl;
//    }

    KFMBitReversalPermutation::PermuteArray< std::complex<double> >(N, index_arr, arr);

    //do the radix-2 FFT
    KFMFastFourierTransformUtilities::FFTRadixTwo_DIT(N, (double*)&(arr[0]), (double*) &(twiddle[0]) );

    kfmout<<"DFT'd array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        kfmout<<arr[i]<<kfmendl;
    }

    //now we'll do the inverse transform

    //conjugate the twiddle factors
    for(unsigned int i=0; i<N; i++)
    {
        twiddle[i] = std::conj(twiddle[i]);
    }

    //do the radix-2 FFT
    KFMFastFourierTransformUtilities::FFTRadixTwo_DIF(N, (double*)&(arr[0]), (double*) &(twiddle[0]) );

    KFMBitReversalPermutation::PermuteArray< std::complex<double> >(N, index_arr, arr);

    //normalize
    for(unsigned int i=0; i<N; i++)
    {
        arr[i] *= 1.0/((double)N);
    }

    kfmout<<"difference between original and IDFT of the DFT'd array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        kfmout<< arr_orig[i] - arr[i]<<kfmendl;
    }



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////



    std::cout<<"---------------------------------------------------------------"<<std::endl;
    std::cout<<"testing DIF followed by DIT"<<std::endl;

    //fill up the array with a signal
    kfmout<<"Original array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        arr[i] = std::complex<double>( i , 0);
        arr_orig[i] = arr[i];
        kfmout<<arr[i]<<kfmendl;
    }

    //compute twiddle factors
    KFMFastFourierTransformUtilities::ComputeTwiddleFactors(N,twiddle);

//    kfmout<<"twiddle factors = "<<kfmendl;
//    for(unsigned int i=0; i<N/2; i++)
//    {
//        kfmout<<"t("<<i<<") = "<<twiddle[i]<<kfmendl;
//    }

    //do the radix-2 FFT
    KFMFastFourierTransformUtilities::FFTRadixTwo_DIF(N, (double*)&(arr[0]), (double*) &(twiddle[0]) );

    kfmout<<"DFT'd array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        kfmout<<arr[i]<<kfmendl;
    }

    //now we'll do the inverse transform

    //conjugate the twiddle factors
    for(unsigned int i=0; i<N; i++)
    {
        twiddle[i] = std::conj(twiddle[i]);
    }

    //do the radix-2 FFT
    KFMFastFourierTransformUtilities::FFTRadixTwo_DIT(N, (double*)&(arr[0]), (double*) &(twiddle[0]) );

    //normalize
    for(unsigned int i=0; i<N; i++)
    {
        arr[i] *= 1.0/((double)N);
    }

    kfmout<<"difference between original and IDFT of the DFT'd array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        kfmout<< arr_orig[i] - arr[i]<<kfmendl;
    }



    return 0;
}

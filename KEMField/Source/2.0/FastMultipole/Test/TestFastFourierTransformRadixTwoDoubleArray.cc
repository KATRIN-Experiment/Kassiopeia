#include <iostream>
#include <cmath>
#include <iomanip>

#include "KFMBitReversalPermutation.hh"
#include "KFMFastFourierTransformUtilities.hh"
#include "KFMMessaging.hh"

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int N = 128*1024;

    kfmout<<"next biggest power of two after 187 is "<<KFMBitReversalPermutation::NextLowestPowerOfTwo(187)<<kfmendl;
    kfmout<<"next biggest power of two after 2345 is "<<KFMBitReversalPermutation::NextLowestPowerOfTwo(2345)<<kfmendl;
    kfmout<<"next biggest power of two after 78456 is "<<KFMBitReversalPermutation::NextLowestPowerOfTwo(78456)<<kfmendl;

    const unsigned int stride = 2;
    unsigned int index_arr[N];
    double arr[stride*N];
    double arr_orig[stride*N];
    double twiddle[N];

    KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(N,index_arr);

    //fill up the array with a signal
    kfmout<<"Original array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        arr[i*stride] = i%4;
        arr[i*stride + 1] = 0.0;
        arr_orig[i*stride] = arr[i*stride];
        arr_orig[i*stride + 1] = arr_orig[i*stride + 1];
        kfmout<<arr[i*stride]<<", "<<arr[i*stride + 1]<<kfmendl;
    }

    //compute twiddle factors
    KFMFastFourierTransformUtilities::ComputeTwiddleFactors(N,twiddle);

    kfmout<<"twiddle factors = "<<kfmendl;
    for(unsigned int i=0; i<N/2; i++)
    {
        kfmout<<"t("<<i<<") = "<<twiddle[i*stride]<<", "<<twiddle[i*stride + 1]<<kfmendl;
    }

    //perform bit reversed address permutation
    KFMBitReversalPermutation::PermuteArray< double >(N, stride, index_arr, arr);

    //do the radix-2 FFT
    KFMFastFourierTransformUtilities::FFTRadixTwo(N, arr, twiddle);

    kfmout<<"DFT'd array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        kfmout<<arr[i*stride]<<", "<<arr[i*stride + 1]<<kfmendl;
    }

    //now we'll do the inverse transform

    //conjugate the input
    for(unsigned int i=0; i<N; i++)
    {
        arr[i*stride + 1] = -1*arr[i*stride + 1];
    }

    //perform bit reversed address permutation
    KFMBitReversalPermutation::PermuteArray< double >(N, stride, index_arr, arr);

    //do the radix-2 FFT
    KFMFastFourierTransformUtilities::FFTRadixTwo(N, arr, twiddle);

    //conjugate the output
    for(unsigned int i=0; i<N; i++)
    {
        arr[i*stride + 1] = -1*arr[i*stride + 1];
    }

    //normalize
    for(unsigned int i=0; i<N; i++)
    {
        arr[i*stride] *= 1.0/((double)N);
        arr[i*stride + 1] *= 1.0/((double)N);
    }

    kfmout<<"difference between original and IDFT of the DFT'd array = "<<kfmendl;
    for(unsigned int i=0; i<N; i++)
    {
        kfmout<<arr_orig[i*stride] - arr[i*stride]<<", "<<arr_orig[i*stride+1] - arr[i*stride+1]<<kfmendl;
    }

    return 0;
}

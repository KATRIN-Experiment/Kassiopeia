#include "KFMBitReversalPermutation.hh"
#include "KFMFastFourierTransformUtilities.hh"
#include "KFMMessaging.hh"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int N = 27;

    unsigned int index_arr[N];
    std::complex<double> arr[N];
    std::complex<double> arr_orig[N];
    std::complex<double> twiddle[N];

    KFMBitReversalPermutation::ComputeBitReversedIndices(N, 3, index_arr);

    //fill up the array with a signal
    kfmout << "Original array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        if (i < 5) {
            arr[i] = std::complex<double>(i, 0);
        }
        else {
            arr[i] = 0;
        }
        arr_orig[i] = arr[i];
        kfmout << arr[i] << kfmendl;
    }

    //compute twiddle factors
    KFMFastFourierTransformUtilities::ComputeTwiddleFactors(N, twiddle);

    kfmout << "twiddle factors = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << "t(" << i << ") = " << twiddle[i] << kfmendl;
    }

    //    //perform bit reversed address permutation
    KFMBitReversalPermutation::PermuteArray<std::complex<double>>(N, index_arr, arr);

    //do the radix-3 FFT
    KFMFastFourierTransformUtilities::FFTRadixThree(N, arr, twiddle);

    kfmout << "DFT'd array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << arr[i] << kfmendl;
    }

    //now we'll do the inverse transform

    //    //conjugate the twiddle factors
    //    for(unsigned int i=0; i<N; i++)
    //    {
    //        twiddle[i] = std::conj(twiddle[i]);
    //    }

    //conjugate the input
    for (unsigned int i = 0; i < N; i++) {
        arr[i] = std::conj(arr[i]);
    }


    //  //perform bit reversed address permutation
    KFMBitReversalPermutation::PermuteArray<std::complex<double>>(N, index_arr, arr);


    //do the radix-3 FFT
    KFMFastFourierTransformUtilities::FFTRadixThree(N, arr, twiddle);

    //conjugate the output
    for (unsigned int i = 0; i < N; i++) {
        arr[i] = std::conj(arr[i]);
    }


    //normalize
    for (unsigned int i = 0; i < N; i++) {
        arr[i] *= 1.0 / ((double) N);
    }

    kfmout << "IDFT of the DFT'd array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << arr[i] << kfmendl;
    }


    kfmout << "difference between original and IDFT of the DFT'd array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << arr[i] - arr_orig[i] << kfmendl;
    }


    return 0;
}

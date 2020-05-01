#include "KFMBitReversalPermutation.hh"
#include "KFMFastFourierTransformUtilities.hh"
#include "KFMMessaging.hh"

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{
    const unsigned int N = 8;
    const unsigned int M = KFMFastFourierTransformUtilities::ComputeBluesteinArraySize(N);

    kfmout << "N = " << N << " and M = " << M << kfmendl;

    unsigned int permutation[M];
    std::complex<double>* data = new std::complex<double>[N];
    std::complex<double>* original_data = new std::complex<double>[N];
    std::complex<double>* scale = new std::complex<double>[N];
    std::complex<double>* twiddle = new std::complex<double>[M];
    std::complex<double>* conj_twiddle = new std::complex<double>[M];
    std::complex<double>* circulant = new std::complex<double>[M];
    std::complex<double>* workspace = new std::complex<double>[M];

    //compute bit reversed address permutation
    KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(M, permutation);

    //fill up the array with a signal
    kfmout << "Original array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        data[i] = std::complex<double>(i % N, 0);
        original_data[i] = data[i];
        kfmout << data[i] << kfmendl;
    }

    //compute the scale factors
    KFMFastFourierTransformUtilities::ComputeBluesteinScaleFactors(N, scale);

    //compute the twiddle factors
    KFMFastFourierTransformUtilities::ComputeTwiddleFactors(M, twiddle);

    //compute the conjugate twiddle factors
    KFMFastFourierTransformUtilities::ComputeConjugateTwiddleFactors(M, conj_twiddle);

    //compute the circulant
    KFMFastFourierTransformUtilities::ComputeBluesteinCirculantVector(N, M, twiddle, scale, circulant);

    //now compute the FFT with Bluestein algorithm
    KFMFastFourierTransformUtilities::FFTBluestein(N, M, data, twiddle, conj_twiddle, scale, circulant, workspace);

    std::cout << "scale = " << std::endl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << scale[i] << kfmendl;
    }

    std::cout << "circulant = " << std::endl;
    for (unsigned int i = 0; i < M; i++) {
        kfmout << circulant[i] << kfmendl;
    }


    kfmout << "DFT'd array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << data[i] << kfmendl;
    }

    //now we'll do the inverse transform
    //conjugate the input
    for (unsigned int i = 0; i < N; i++) {
        data[i] = std::conj(data[i]);
    }

    //now compute the FFT with Bluestein algorithm
    KFMFastFourierTransformUtilities::FFTBluestein(N, M, data, twiddle, conj_twiddle, scale, circulant, workspace);

    //conjugate the output
    for (unsigned int i = 0; i < N; i++) {
        data[i] = std::conj(data[i]);
    }

    //normalize
    for (unsigned int i = 0; i < N; i++) {
        data[i] *= 1.0 / ((double) N);
    }

    kfmout << "IDFT of the DFT'd array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << data[i] << kfmendl;
    }

    kfmout << "difference between original and IDFT of the DFT'd array = " << kfmendl;
    for (unsigned int i = 0; i < N; i++) {
        kfmout << original_data[i] - data[i] << kfmendl;
    }


    delete[] data;
    delete[] original_data;
    delete[] scale;
    delete[] twiddle;
    delete[] circulant;
    delete[] workspace;

    return 0;
}

#include "KFMFastFourierTransformUtilities.hh"
#include "KFMMessaging.hh"

#include <cmath>

#define SQRT_THREE_OVER_TWO 0.866025403784438646763723170753

namespace KEMField
{

////////////////////////////////////////////////////////////////////////////////
//TWIDDLE FACTORS

void
KFMFastFourierTransformUtilities::ComputeTwiddleFactors(unsigned int N, std::complex<double>* twiddle)
{

    double di, dN;
    dN = N;
    for(unsigned int i=0; i<N; i++)
    {
        di = i;
        twiddle[i] = std::complex<double>(std::cos((2.0*M_PI*di)/dN), std::sin((2.0*M_PI*di)/dN));
    }

//    //This method is from:
//    //An Algorithm for Computing the Mixed Radix Fast Fourier Transform
//    //Richard C. Singleton
//    //IEEE Transactions on Audio and Electroacoustics Vol. Au-17 No. 2 June 1969

//    //intial values needed for recursion
//    double theta = 2.0*M_PI/((double)N);
//    double sin_theta = std::sin(theta);
//    double sin_theta_over_two = std::sin(theta/2.0);
//    double eta_real = -2.0*sin_theta_over_two*sin_theta_over_two;
//    double eta_imag = sin_theta;

//    //space to store last value
//    double real = 1.0;
//    double imag = 0.0;

//    //init recursion
//    twiddle[0] = std::complex<double>(real,imag);

//    //scratch space
//    double a, b, mag2, delta;

//    for(unsigned int i=1; i<N; i++)
//    {
//        a = real + eta_real*real - eta_imag*imag;
//        b = imag + eta_imag*real + eta_real*imag;

//        //re-scale to correct round off errors
//        mag2 = a*a + b*b;
//        delta = 1.0/std::sqrt(mag2);

//        a *= delta;
//        b *= delta;

//        real = a;
//        imag = b;

//        twiddle[i] = std::complex<double>(real, imag);
//    }
}


void
KFMFastFourierTransformUtilities::ComputeConjugateTwiddleFactors(unsigned int N, std::complex<double>* conj_twiddle)
{
    //This method is from:
    //An Algorithm for Computing the Mixed Radix Fast Fourier Transform
    //Richard C. Singleton
    //IEEE Transactions on Audio and Electroacoustics Vol. Au-17 No. 2 June 1969

    //intial values needed for recursion
    double theta = 2.0*M_PI/((double)N);
    double sin_theta = std::sin(theta);
    double sin_theta_over_two = std::sin(theta/2.0);
    double eta_real = -2.0*sin_theta_over_two*sin_theta_over_two;
    double eta_imag = sin_theta;

    //space to store last value
    double real = 1.0;
    double imag = 0.0;

    //init recursion
    conj_twiddle[0] = std::complex<double>(real, -1.0*imag);

    //scratch space
    double a, b, mag2, delta;

    for(unsigned int i=1; i<N; i++)
    {
        a = real + eta_real*real - eta_imag*imag;
        b = imag + eta_imag*real + eta_real*imag;

        //re-scale to correct round off errors
        mag2 = a*a + b*b;
        delta = 1.0/std::sqrt(mag2);

        a *= delta;
        b *= delta;

        real = a;
        imag = b;

        conj_twiddle[i] = std::complex<double>(real, -1.0*imag);
    }
}

////////////////////////////////////////////////////////////////////////////////
//RADIX-2

void
KFMFastFourierTransformUtilities::FFTRadixTwo_DIT(unsigned int N, std::complex<double>* data, std::complex<double>* twiddle)
{
    FFTRadixTwo_DIT(N, (double*)&(data[0]), (double*) &(twiddle[0]) );
}

void
KFMFastFourierTransformUtilities::FFTRadixTwo_DIF(unsigned int N, std::complex<double>* data, std::complex<double>* twiddle)
{
    FFTRadixTwo_DIF(N, (double*)&(data[0]), (double*) &(twiddle[0]) );
}

void
KFMFastFourierTransformUtilities::ButterflyRadixTwo_CooleyTukey(double* H0, double* H1, double* W)
{
    ////////////////////////////////////////////////////////////////////////
    //See page 23
    //"Inside the FFT Black Box", E. Chu and A. George, Ch. 13, CRC Press, 2000

    //H0 is the element from the even indexed array
    //H1 is the element from the odd index array
    //W is the twiddle factor

    //multiply H1 by the twiddle factor to get W*H1 and store in H1
    double alpha_r = W[0]*H1[0] - W[1]*H1[1];
    double alpha_i = W[1]*H1[0] + W[0]*H1[1];

    H1[0] = H0[0] - alpha_r;
    H1[1] = H0[1] - alpha_i;

    H0[0] = H0[0] + alpha_r;
    H0[1] = H0[1] + alpha_i;
}


void
KFMFastFourierTransformUtilities::ButterflyRadixTwo_GentlemanSande(double* H0, double* H1, double* W)
{
    ////////////////////////////////////////////////////////////////////////
    //See page 25
    //"Inside the FFT Black Box", E. Chu and A. George, Ch. 13, CRC Press, 2000

    //H0 is the element from the even indexed array
    //H1 is the element from the odd index array
    //W is the twiddle factor

    //cache H1
    double h1_r = H1[0];
    double h1_i = H1[1];

    //compute new h1
    H1[0] = H0[0] - h1_r;
    H1[1] = H0[1] - h1_i;

    //compute new n0
    H0[0] += h1_r;
    H0[1] += h1_i;

    //multiply new h1 by twiddle factor
    double alpha_r = W[0]*H1[0] - W[1]*H1[1];
    double alpha_i = W[1]*H1[0] + W[0]*H1[1];

    H1[0] = alpha_r;
    H1[1] = alpha_i;
}

void
KFMFastFourierTransformUtilities::FFTRadixTwo_DIT(unsigned int N, double* data, double* twiddle)
{
    //decimation in time

    //input: data array in bit-address permutated order
    //output: fft of data in normal order

    if(KFMBitReversalPermutation::IsPowerOfTwo(N) )
    {
        unsigned int logN = KFMBitReversalPermutation::LogBaseTwo(N);

        unsigned int butterfly_width;
        unsigned int n_butterfly_groups;
        unsigned int group_start;
        unsigned int butterfly_index;

        for(unsigned int stage = 0; stage < logN; stage++)
        {
            //compute the width of each butterfly
            butterfly_width = KFMBitReversalPermutation::TwoToThePowerOf(stage);

            //compute the number of butterfly groups
            n_butterfly_groups = N/(2*butterfly_width);

            for(unsigned int n = 0; n < n_butterfly_groups; n++)
            {
                //compute the starting index of this butterfly group
                group_start = 2*n*butterfly_width;

                for(unsigned int k=0; k < butterfly_width; k++)
                {
                    butterfly_index = group_start + k; //index

                    ButterflyRadixTwo_CooleyTukey( &(data[2*butterfly_index]),
                                                   &(data[2*butterfly_index + 2*butterfly_width]),
                                                   &(twiddle[2*n_butterfly_groups*k]) );
                }
            }

        }
    }
    else
    {
        kfmout<<"KFMFastFourierTransformUtilities::FFTRadixTwo_DIT: error, array has length: "<<N<<" which is not an integer power of 2."<<kfmendl;
    }
}


void
KFMFastFourierTransformUtilities::FFTRadixTwo_DIF(unsigned int N, double* data, double* twiddle)
{
    //decimation in frequency

    //input: data array in normal order
    //output: fft of data in bit-address permutated order

    if(KFMBitReversalPermutation::IsPowerOfTwo(N) )
    {
        unsigned int logN = KFMBitReversalPermutation::LogBaseTwo(N);

        unsigned int butterfly_width;
        unsigned int n_butterfly_groups;
        unsigned int group_start;
        unsigned int butterfly_index;

        for(unsigned int stage = 0; stage < logN; stage++)
        {
            //compute the number of butterfly groups
            n_butterfly_groups= KFMBitReversalPermutation::TwoToThePowerOf(stage);

            //compute the width of each butterfly
            butterfly_width =  N/(2*n_butterfly_groups);

            for(unsigned int n = 0; n < n_butterfly_groups; n++)
            {
                //compute the starting index of this butterfly group
                group_start = 2*n*butterfly_width;

                for(unsigned int k=0; k < butterfly_width; k++)
                {
                    butterfly_index = group_start + k; //index

                    ButterflyRadixTwo_GentlemanSande( &(data[2*butterfly_index]),
                                                      &(data[2*butterfly_index + 2*butterfly_width]),
                                                      &(twiddle[2*n_butterfly_groups*k]) );
                }
            }

        }
    }
    else
    {
        kfmout<<"KFMFastFourierTransformUtilities::FFTRadixTwo_DIF: error, array has length: "<<N<<" which is not an integer power of 2."<<kfmendl;
    }

}





////////////////////////////////////////////////////////////////////////////////
//RADIX-3

void
KFMFastFourierTransformUtilities::ButterflyRadixThree(std::complex<double>& H0,
                                                      std::complex<double>& H1,
                                                      std::complex<double>& H2,
                                                      std::complex<double>& W0,
                                                      std::complex<double>& W1)
{
    std::complex<double> A = H0;
    std::complex<double> B = W0*H1;
    std::complex<double> C = W1*H2;
    std::complex<double> X1(-0.5, SQRT_THREE_OVER_TWO); //e^(i*(2*pi/3));
    std::complex<double> X2(-0.5, -SQRT_THREE_OVER_TWO); //e^(i*(4*pi/3));

    H0 = A + B + C;
    H1 = A + X1*B + X2*C;
    H2 = A + X2*B + X1*C;
}

void
KFMFastFourierTransformUtilities::FFTRadixThree(unsigned int N, std::complex<double>* data, std::complex<double>* twiddle)
{

    if(KFMBitReversalPermutation::IsPowerOfBase(N,3) )
    {
        unsigned int logN = KFMBitReversalPermutation::LogBaseB(N,3);

        unsigned int butterfly_width;
        unsigned int n_butterfly_groups;
        unsigned int group_start;
        unsigned int butterfly_index;

        for(unsigned int stage = 0; stage < logN; stage++)
        {
            //compute the width of each butterfly
            butterfly_width = KFMBitReversalPermutation::RaiseBaseToThePower(3,stage);

            //compute the number of butterfly groups
            n_butterfly_groups = N/(3*butterfly_width);

            for(unsigned int n = 0; n < n_butterfly_groups; n++)
            {
                //compute the starting index of this butterfly group
                group_start = 3*n*butterfly_width;

                for(unsigned int k=0; k < butterfly_width; k++)
                {
                    butterfly_index = group_start + k; //index

                    ButterflyRadixThree( data[butterfly_index],
                                         data[butterfly_index + butterfly_width],
                                         data[butterfly_index + 2*butterfly_width],
                                         twiddle[n_butterfly_groups*k],
                                         twiddle[2*n_butterfly_groups*k]);
                }
            }
        }
    }
    else
    {
        kfmout<<"KFMFastFourierTransformUtilities::FFTRadixThree: error, array has length: "<<N<<" which is not an integer power of 3."<<kfmendl;
    }


}


////////////////////////////////////////////////////////////////////////////////
//Bluestein Algorithm


unsigned int
KFMFastFourierTransformUtilities::ComputeBluesteinArraySize(unsigned int N)
{
    unsigned int M = 2*(N - 1);
    if( KFMBitReversalPermutation::IsPowerOfTwo(M))
    {
        return M;
    }
    else
    {
        return KFMBitReversalPermutation::NextLowestPowerOfTwo(M);
    }
}


void
KFMFastFourierTransformUtilities::ComputeBluesteinScaleFactors(unsigned int N, std::complex<double>* scale)
{
    //STEP A
    double theta = M_PI/((double)N);
    unsigned int i2;
    double x;

    for(unsigned int i=0; i<N; i++)
    {
        i2 = i*i % (2*N); //taking the modulus here results in a more accurate DFT/IDFT
        x = theta*i2;
        scale[i] = std::complex<double>( std::cos(x), std::sin(x) );
    }

    //IMPORTANT NOTE!
    //This function computes the CONJUGATE of the scale factors as
    //defined by equation 13.22, page 127 of
    //"Inside the FFT Black Box", E. Chu and A. George, Ch. 13, CRC Press, 2000
    //we do this so we can avoid having to take a complex conjugate of the scale
    //factors when performing the actual FFT
}

void
KFMFastFourierTransformUtilities::ComputeBluesteinCirculantVector(
unsigned int N,
unsigned int M,
std::complex<double>* twiddle, //must be size M
std::complex<double>* scale, //must be size N
std::complex<double>* circulant) //must be size M
{
    //STEP B
    unsigned int mid = M - N + 1;

    for(unsigned int i=0; i<N; i++)
    {
        //we take the conjugate here because the way we have computed the scale factors
        circulant[i] = std::conj(scale[i]);
    }

    for(unsigned int i=mid; i<M; i++)
    {
        //we take the conjugate here because the way we have computed the scale factors
        circulant[i] = std::conj(scale[M-i]);
    }

    for(unsigned int i=N; i<mid; i++)
    {
        circulant[i] = 0.0;
    }

    //STEP C
    //now we perform an in-place DFT on the circulant vector

    //expects normal order input, produces bit-address permutated output
    KFMFastFourierTransformUtilities::FFTRadixTwo_DIF(M, circulant, twiddle);
}

void
KFMFastFourierTransformUtilities::FFTBluestein(
unsigned int N,
unsigned int M,
std::complex<double>* data, //must be size N
std::complex<double>* twiddle, //must be size M
std::complex<double>* conj_twiddle, //must be size M
std::complex<double>* scale, //must be size N
std::complex<double>* circulant, //must be size M
std::complex<double>* workspace) //must be size M
{

    //STEP D
    //copy the data into the workspace and scale by the scale factor
    for(unsigned int i=0; i<N; i++)
    {
        workspace[i] = data[i]*scale[i];
    }

    //fill out the rest of the extended vector with zeros
    for(unsigned int i=N; i<M; i++)
    {
        workspace[i] = std::complex<double>(0.0,0.0);
    }

    //STEP E
    //perform the DFT on the workspace
    //do radix-2 FFT w/ decimation in frequency (normal order input, bit-address permutated output)
    KFMFastFourierTransformUtilities::FFTRadixTwo_DIF(M, (double*)(&(workspace[0])), (double*)(&(twiddle[0])) );

    //STEP F
    //now we scale the workspace with the circulant vector
    for(unsigned int i=0; i<M; i++)
    {
        workspace[i] *= circulant[i];
    }

    //STEP G
    //now perform the inverse DFT on the workspace
    //do radix-2 FFT w/ decimation in time (bit-address permutated input, normal order output)
    KFMFastFourierTransformUtilities::FFTRadixTwo_DIT(M, (double*)(&(workspace[0])), (double*)(&(conj_twiddle[0])) );

    //STEP H
    //renormalize to complete IDFT, extract and scale at the same time
    double norm = 1.0/((double)M);
    for(unsigned int i=0; i<N; i++)
    {
        data[i] = norm*workspace[i]*scale[i];
    }

}


}

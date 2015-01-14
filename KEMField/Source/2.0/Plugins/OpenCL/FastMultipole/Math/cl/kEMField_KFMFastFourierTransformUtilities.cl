#ifndef KFMFastFourierUtilities_Defined_H
#define KFMFastFourierUtilities_Defined_H

#include "kEMField_defines.h"

////////////////////////////////////////////////////////////////////////////////
//base 2 integer log and exponentiation

unsigned int
LogBaseTwo(unsigned int N)
{
    //taken from Bit Twiddling Hacks
    //http://graphics.stanford.edu/~seander/bithacks.html
    unsigned int p = 0;
    while (N >>= 1)
    {
        p++;
    }
    return p;
}


unsigned int
TwoToThePowerOf(unsigned int N)
{
    unsigned int val = 1;
    for(unsigned int i=0; i<N; i++)
    {
        val *= 2;
    }
    return val;
}


bool
IsPowerOfBase(unsigned int N, unsigned int B)
{
    //check if N is a perfect power of B, this is very slow!!
    if(N < B)
    {
        //N = 1 = B^0; is not considered to be a power in this case
        return false;
    }
    else
    {
        unsigned int i = 1;
        while(i < N)
        {
            i *= B;
        }

        if(N == i){return true;}
        return false;
    }
}

unsigned int
NextLowestPowerOfTwo(unsigned int N)
{
    if(IsPowerOfBase(N,2) )
    {
        return N;
    }
    else
    {
        unsigned p = LogBaseTwo(N);
        return TwoToThePowerOf(p+1);
    }
}

unsigned int
BluesteinArraySize(unsigned int N)
{
    unsigned int M = 2*(N - 1);
    if(IsPowerOfBase(M,2)){return M;};
    return NextLowestPowerOfTwo(M);
}


////////////////////////////////////////////////////////////////////////////////
//array permutation
void
PermuteArray(unsigned int N, __constant const unsigned int* permutation_index_arr, CL_TYPE2* arr)
{
    //expects an array of size N
    CL_TYPE2 val;
    for(unsigned int i=0; i<N; i++)
    {
        unsigned int perm = permutation_index_arr[i];
        if(i < perm )
        {
            //swap values
            val = arr[i];
            arr[i] = arr[perm];
            arr[perm] = val;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//Radix-2 Algorithm

void
FFTRadixTwo_DIT(unsigned int N, CL_TYPE2* data, __constant const CL_TYPE2* twiddle)
{
    //temporary workspace
    CL_TYPE2 H0;
    CL_TYPE2 H1;
    CL_TYPE2 W;
    CL_TYPE2 Z;

    unsigned int logN = LogBaseTwo(N);
    unsigned int butterfly_width;
    unsigned int n_butterfly_groups;
    unsigned int group_start;
    unsigned int butterfly_index;

    for(unsigned int stage = 0; stage < logN; stage++)
    {
        //compute the width of each butterfly
        butterfly_width = TwoToThePowerOf(stage);

        //compute the number of butterfly groups
        n_butterfly_groups = N/(2*butterfly_width);

        for(unsigned int n = 0; n < n_butterfly_groups; n++)
        {
            //compute the starting index of this butterfly group
            group_start = 2*n*butterfly_width;

            for(unsigned int k=0; k < butterfly_width; k++)
            {
                butterfly_index = group_start + k; //index

                H0 = data[butterfly_index];
                H1 = data[butterfly_index + butterfly_width];
                W = twiddle[n_butterfly_groups*k];

                //here we use the Cooly-Tukey butterfly

                //multiply H1 by twiddle factor to get W*H1, store temporary workspace Z
                Z.s0 = (H1.s0)*(W.s0) - (H1.s1)*(W.s1);
                Z.s1 = (H1.s0)*(W.s1) + (H1.s1)*(W.s0);

                //compute the update
                //H0' = H0 + W*H1
                //H1' = H0 - W*H1
                H1 = H0;

                H0 += Z;
                H1 -= Z;

                data[butterfly_index] = H0;
                data[butterfly_index + butterfly_width] = H1;
            }
        }
    }
}


void
FFTRadixTwo_DIF(unsigned int N, CL_TYPE2* data, __constant const CL_TYPE2* twiddle)
{
    //temporary workspace
    CL_TYPE2 H0;
    CL_TYPE2 H1;
    CL_TYPE2 W;
    CL_TYPE2 Z;

    unsigned int logN = LogBaseTwo(N);
    unsigned int butterfly_width;
    unsigned int n_butterfly_groups;
    unsigned int group_start;
    unsigned int butterfly_index;

    for(unsigned int stage = 0; stage < logN; stage++)
    {
        //compute the number of butterfly groups
        n_butterfly_groups = TwoToThePowerOf(stage);

        //compute the width of each butterfly
        butterfly_width = N/(2*n_butterfly_groups);

        for(unsigned int n = 0; n < n_butterfly_groups; n++)
        {
            //compute the starting index of this butterfly group
            group_start = 2*n*butterfly_width;

            for(unsigned int k=0; k < butterfly_width; k++)
            {
                butterfly_index = group_start + k; //index

                H0 = data[butterfly_index];
                H1 = data[butterfly_index + butterfly_width];
                W = twiddle[n_butterfly_groups*k];

                //here we use the Gentleman-Sande butterfly

                //compute the update
                //first cache H1 in Z
                Z = H1;

                //set H1' = H0 - H1
                H1 = H0 - Z;

                //set H0 = H0 + H1
                H0 += Z;

                //multiply H1 by twiddle factor to get W*H1, to obtain H1' = (H0 - H1)*W
                Z.s0 = (H1.s0)*(W.s0) - (H1.s1)*(W.s1);
                Z.s1 = (H1.s0)*(W.s1) + (H1.s1)*(W.s0);

                data[butterfly_index] = H0;
                data[butterfly_index + butterfly_width] = Z;
            }
        }
    }
}




////////////////////////////////////////////////////////////////////////////////
//Bluestein Algorithm

void
FFTBluestein(unsigned int N,
             unsigned int M,
             __constant CL_TYPE2* twiddle, //must be size M
             __constant CL_TYPE2* conj_twiddle, //must be size M
             __constant CL_TYPE2* scale, //must be size N
             __constant CL_TYPE2* circulant, //must be size M
             CL_TYPE2* data) //must be size M, initially filled up to N
{


    //STEP D
    //copy the data into the workspace and scale by the scale factor
    CL_TYPE2 B;
    CL_TYPE2 A;
    CL_TYPE2 Z;

    for(size_t i=0; i<N; i++)
    {
        A = data[i];
        B = scale[i];
        Z.s0 = (A.s0)*(B.s0) - (A.s1)*(B.s1);
        Z.s1 =  (A.s1)*(B.s0) + (A.s0)*(B.s1);
        data[i] = Z;
    }

    //fill out the rest of the extended vector with zeros
    Z = 0.0;
    for(size_t i=N; i<M; i++)
    {
        data[i] = Z;
    }

    //do a decimation in frequency radix-2 FFT
    FFTRadixTwo_DIF(M, data, twiddle);

    //STEP F
    //now we scale the workspace with the circulant vector, and conjugate for input to dft
    for(size_t i=0; i<M; i++)
    {
        A = data[i];
        B = circulant[i];
        Z.s0 = (A.s0)*(B.s0) - (A.s1)*(B.s1);
        Z.s1 = ( (A.s0)*(B.s1) + (A.s1)*(B.s0) );
        data[i] = Z;
    }

    //do a decimation in time radix-2 inverse FFT
    FFTRadixTwo_DIT(M, data, conj_twiddle);

    //STEP H
    //renormalize to complete IDFT, extract and scale at the same time
    CL_TYPE norm = 1.0/((CL_TYPE)M);
    for(size_t i=0; i<N; i++)
    {
        A = data[i];
        B = scale[i];

        Z.s0 = (A.s0)*(B.s0) - (A.s1)*(B.s1);
        Z.s1 = (A.s0)*(B.s1) + (A.s1)*(B.s0);
        data[i] = norm*Z;
    }
}


#endif /* KFMFastFourierUtilities_Defined_H */

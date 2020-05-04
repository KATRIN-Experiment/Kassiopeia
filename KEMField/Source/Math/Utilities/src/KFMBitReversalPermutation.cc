#include "KFMBitReversalPermutation.hh"

#include "KFMMessaging.hh"


namespace KEMField
{


bool KFMBitReversalPermutation::IsPowerOfTwo(unsigned int N)
{
    //taken from Bit Twiddling Hacks
    //http://graphics.stanford.edu/~seander/bithacks.html
    return N && !(N & (N - 1));
}


unsigned int KFMBitReversalPermutation::LogBaseTwo(unsigned int N)
{
    //taken from Bit Twiddling Hacks
    //http://graphics.stanford.edu/~seander/bithacks.html
    unsigned int p = 0;
    while (N >>= 1) {
        p++;
    }
    return p;
}


unsigned int KFMBitReversalPermutation::TwoToThePowerOf(unsigned int N)
{
    unsigned int val = 1;
    for (unsigned int i = 0; i < N; i++) {
        val *= 2;
    }
    return val;
}

unsigned int KFMBitReversalPermutation::NextLowestPowerOfTwo(unsigned int N)
{
    if (IsPowerOfTwo(N)) {
        return N;
    }
    else {
        unsigned int p = LogBaseTwo(N);
        return TwoToThePowerOf(p + 1);
    }
}


bool KFMBitReversalPermutation::IsPowerOfBase(unsigned int N, unsigned int B)
{
    //check if N is a perfect power of B, this is very slow!!
    if (N < B) {
        return false;
    }
    else {
        unsigned int i = 1;
        while (i < N) {
            i *= B;
        }

        if (N == i) {
            return true;
        }
        return false;
    }
}

unsigned int KFMBitReversalPermutation::RaiseBaseToThePower(unsigned int B, unsigned int N)
{
    unsigned int val = 1;
    for (unsigned int i = 0; i < N; i++) {
        val *= B;
    }
    return val;
}

unsigned int KFMBitReversalPermutation::LogBaseB(unsigned int N, unsigned int B)
{
    //we assume that N is a perfect power of B
    //but if not we return the leading power
    if (N != 0) {
        if (N == 1) {
            return 0;
        }

        unsigned int power = 0;
        unsigned int quotient = N;

        do {
            quotient /= B;
            power++;
        } while (quotient > 1);
        return power;
    }
    else {
        //error
        return 0;
    }
}


//void
//KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(unsigned int N, unsigned int* index_arr)
//{
//    //this function uses the recursive Buneman algorithm to compute
//    //the bit reversed permutation of an array of length N = 2^p with entries 0,1,2...N-1
//    //the permutated indices are stored in index_arr
//    //this is slow but simple
//    //for details see
//    //Fast Fourier Transforms by James S. Walker, CRC Press


//    if( IsPowerOfTwo(N) && N != 0)
//    {
//        unsigned int p = LogBaseTwo(N);

//        if(N == 1) //p = 0
//        {
//            index_arr[0] = 0;
//            return;
//        }

//        if(N == 2) //p = 1
//        {
//            index_arr[0] = 0;
//            index_arr[1] = 1;
//            return;
//        }

//        index_arr[0] = 0;
//        index_arr[1] = 1;

//        unsigned int mid;
//        for(unsigned int r=2; r<=p; r++)
//        {
//            mid = TwoToThePowerOf(r-1);
//            for(unsigned int q=0; q<mid; q++)
//            {
//                index_arr[q] *= 2;
//                index_arr[q + mid] = index_arr[q] + 1;
//            }
//        }
//    }
//    else
//    {
//        kfmout<<"KFMBitReversalPermutation::ComputeBitReversedIndices: error, called with non-power of two array size."<<kfmendl;
//        kfmexit(1);
//        //error
//    }
//}


void KFMBitReversalPermutation::ComputeBitReversedIndicesBaseTwo(unsigned int N, unsigned int* index_arr)
{
    //this function uses the recursive Buneman algorithm to compute
    //the bit reversed permutation of an array of length N = 2^p with entries 0,1,2...N-1
    //the permutated indices are stored in index_arr
    //this is slow but simple
    //for details see
    //Fast Fourier Transforms by James S. Walker, CRC Press


    if (IsPowerOfTwo(N) && N != 0) {
        unsigned int p = LogBaseTwo(N);

        if (N == 1)  //p = 0
        {
            index_arr[0] = 0;
            return;
        }

        if (N == 2)  //p = 1
        {
            index_arr[0] = 0;
            index_arr[1] = 1;
            return;
        }

        index_arr[0] = 0;
        index_arr[1] = 1;

        unsigned int mid;
        for (unsigned int r = 2; r <= p; r++) {
            mid = TwoToThePowerOf(r - 1);
            for (unsigned int q = 0; q < mid; q++) {
                index_arr[q] *= 2;
                index_arr[q + mid] = index_arr[q] + 1;
            }
        }
    }
    else {
        kfmout
            << "KFMBitReversalPermutation::ComputeBitReversedIndices: error, called with non-power of two array size."
            << kfmendl;
        kfmexit(1);
        //error
    }
}

void KFMBitReversalPermutation::ComputeBitReversedIndices(unsigned int N, unsigned int B, unsigned int* index_arr)
{
    //this function is the base B extention of the recursive Buneman algorithm to compute
    //the bit reversed permutation of an array of length N

    if (IsPowerOfBase(N, B) && N != 0) {
        unsigned int p = LogBaseB(N, B);

        if (N == 1)  //p = 0
        {
            index_arr[0] = 0;
            return;
        }

        //p >= 1
        for (unsigned int i = 0; i < B; i++) {
            index_arr[i] = i;
        }

        if (N == B) {
            return;
        };

        //p >=2
        unsigned int division;
        for (unsigned int r = 2; r <= p; r++) {
            division = RaiseBaseToThePower(B, r - 1);
            for (unsigned int q = 0; q < division; q++) {
                index_arr[q] *= B;
                for (unsigned int s = 1; s < B; s++) {
                    index_arr[q + s * division] = index_arr[q] + s;
                }
            }
        }
    }
    else {
        kfmout << "KFMBitReversalPermutation::ComputeBitReversedIndices: error, called with non-power of " << B
               << " array size." << kfmendl;
        kfmexit(1);
        //error
    }
}

bool KFMBitReversalPermutation::Factor(unsigned int N, unsigned int n_factors, unsigned int* factors,
                                       unsigned int* powers)
{
    unsigned int test = 1;
    for (unsigned int i = 0; i < n_factors; i++) {
        unsigned int quotient = N;
        powers[i] = 0;
        while (quotient % factors[i] == 0) {
            quotient /= factors[i];
            powers[i] += 1;
            test *= factors[i];
        }
    }

    if (test == N) {
        return true;
    }

    return false;
}


}  // namespace KEMField

#ifndef KFMBitReversalPermutation_HH__
#define KFMBitReversalPermutation_HH__

#include <cstddef>
#include <vector>

namespace KEMField
{

/*
*
*@file KFMBitReversalPermutation.hh
*@class KFMBitReversalPermutation
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Nov 18 15:47:20 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KFMBitReversalPermutation
{
    public:
        KFMBitReversalPermutation(){};
        virtual ~KFMBitReversalPermutation(){};

        static bool IsPowerOfTwo(unsigned int N);
        static unsigned int TwoToThePowerOf(unsigned int N); //N must be >= 0
        static unsigned int LogBaseTwo(unsigned int N);
        static unsigned int NextLowestPowerOfTwo(unsigned int N);

        //factor the integer N into powers of the factors listed in factors
        static bool Factor(unsigned int N, unsigned int n_factors, unsigned int* factors, unsigned int* powers);

        static bool IsPowerOfBase(unsigned int N, unsigned int B);
        static unsigned int RaiseBaseToThePower(unsigned int B, unsigned int N); //N must be >= 0
        static unsigned int LogBaseB(unsigned int N, unsigned int B);

        //static void ComputeBitReversedIndicesBaseTwo(unsigned int N, unsigned int* index_arr); //must have N = 2^P, with P an integer
        static void ComputeBitReversedIndicesBaseTwo(unsigned int N, unsigned int* index_arr); //must have N = 2^P, with P an integer

        //must have length N = B^P, with P an integer
        //B is the base of the number system used to compute the bit indices
        static void ComputeBitReversedIndices(unsigned int N, unsigned int B, unsigned int* index_arr);


        template<typename DataType >
        static void PermuteArray(unsigned int N, const unsigned int* permutation_index_arr, DataType* arr) //data_size of 1
        {
            //expects an array of size N
            DataType val;
            for(unsigned int i=0; i<N; i++)
            {
                unsigned int perm = permutation_index_arr[i];
                if(i < perm )
                {
                    //swap values
                    val = arr[i];
                    arr[i] = arr[ perm ];
                    arr[perm] = val;
                }
            }
        }

        template<typename DataType >
        static void PermuteArray(unsigned int N, unsigned int data_size, const unsigned int* permutation_index_arr, DataType* arr) //arbitrary data_size
        {
            //expects an array of size N*data_size
            DataType val;
            for(unsigned int i=0; i<N; i++)
            {
                if(i < permutation_index_arr[i] )
                {
                    //swap values
                    unsigned int old_index = i;
                    unsigned int new_index = permutation_index_arr[i];

                    for(unsigned int j=0; j<data_size; j++)
                    {
                        val = arr[old_index*data_size + j];
                        arr[old_index*data_size + j] = arr[new_index*data_size + j];
                        arr[new_index*data_size + j] = val;
                    }
                }
            }
        }


        template<typename DataType >
        static void PermuteVector(unsigned int N, const unsigned int* permutation_index_arr, std::vector<DataType>* arr)
        {
            //expects an array of size N
            DataType val;
            for(unsigned int i=0; i<N; i++)
            {
                unsigned int perm = permutation_index_arr[i];
                if(i < perm )
                {
                    //swap values
                    val = arr[i];
                    arr[i] = arr[ perm ];
                    arr[perm] = val;
                }
            }
        }


    private:
};


}


#endif /* KFMBitReversalPermutation_H__ */

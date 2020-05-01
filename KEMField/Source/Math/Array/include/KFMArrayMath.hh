#ifndef KFMArrayMath_HH__
#define KFMArrayMath_HH__

#include "KFMMessaging.hh"

#include <bitset>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace KEMField
{


/**
*
*@file KFMArrayMath.hh
*@class KFMArrayMath
*@brief collection of math functions used by array library
*@details
*
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue May 28 21:59:09 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

#if defined(__SIZEOF_INT__) && defined(CHAR_BIT)
#define MAX_MORTON_BITS __SIZEOF_INT__* CHAR_BIT
#else
#define MAX_MORTON_BITS 32
#endif

class KFMArrayMath
{
  public:
    KFMArrayMath(){};
    virtual ~KFMArrayMath(){};

    //modulus of two integers
    static unsigned int Modulus(int arg, int n)
    {
        //returns arg mod n;
        double div = ((double) arg) / ((double) n);
        return (unsigned int) (std::fabs((double) arg - std::floor(div) * ((double) n)));
    }

    //for a multidimensional array (using row major indexing) which has the
    //dimensions specified in DimSize, this function computes the offset from
    //the first element given the indices in the array Index
    template<unsigned int NDIM>
    inline static unsigned int OffsetFromRowMajorIndex(const unsigned int* DimSize, const unsigned int* Index)
    {
        unsigned int val = Index[0];
        for (unsigned int i = 1; i < NDIM; i++) {
            val *= DimSize[i];
            val += Index[i];
        }
        return val;
    }

    //for a multidimensional array (using row major indexing) which has the
    //dimensions specified in DimSize, this function computes the stride between
    //consecutive elements in the selected dimension given that the other indices are fixed
    //the first element given the indices in the array Index
    template<unsigned int NDIM>
    inline static unsigned int StrideFromRowMajorIndex(unsigned int selected_dim, const unsigned int* DimSize)
    {
        unsigned int val = 1;
        for (unsigned int i = 0; i < NDIM; i++) {
            if (i > selected_dim) {
                val *= DimSize[i];
            };
        }
        return val;
    }


    //for a multidimensional array (using row major indexing) which has the
    //dimensions specified in DimSize, this function computes the indices of
    //the elements which has the given offset from the first element
    template<unsigned int NDIM>
    inline static void RowMajorIndexFromOffset(unsigned int offset, const unsigned int* DimSize, unsigned int* Index)
    {
        unsigned int div[NDIM];

        //in row major format the last index varies the fastest
        unsigned int i;
        for (unsigned int d = 0; d < NDIM; d++) {
            i = NDIM - d - 1;

            if (d == 0) {
                Index[i] = KFMArrayMath::Modulus(offset, DimSize[i]);
                div[i] = (offset - Index[i]) / DimSize[i];
            }
            else {
                Index[i] = KFMArrayMath::Modulus(div[i + 1], DimSize[i]);
                div[i] = (div[i + 1] - Index[i]) / DimSize[i];
            }
        }
    }

    //checks if all the indices in Index are in the valid range
    template<unsigned int NDIM>
    inline static bool CheckIndexValidity(const unsigned int* DimSize, const unsigned int* Index)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            if (Index[i] >= DimSize[i]) {
                return false;
            };
        }
        return true;
    };


    //given the dimensions of an array, computes its total size, assuming all dimensions are non-zero
    template<unsigned int NDIM> inline static unsigned int TotalArraySize(const unsigned int* DimSize)
    {
        unsigned int val = 1;
        for (unsigned int i = 0; i < NDIM; i++) {
            val *= DimSize[i];
        }
        return val;
    }

    //compute 2^N at compile time
    template<unsigned int N> struct PowerOfTwo
    {
        enum
        {
            value = 2 * PowerOfTwo<N - 1>::value
        };
    };

    //compute integer division at compile time
    template<int numerator, int denominator> struct Divide
    {
        enum
        {
            value = Divide<numerator, 1>::value / Divide<denominator, 1>::value
        };
    };

    template<unsigned int NDIM>
    static void OffsetsForReversedIndices(const unsigned int* DimSize, unsigned int* ReversedIndex)
    {
        unsigned int total_array_size = KFMArrayMath::TotalArraySize<NDIM>(DimSize);
        unsigned int index[NDIM];
        for (unsigned int i = 0; i < total_array_size; i++) {
            KFMArrayMath::RowMajorIndexFromOffset<NDIM>(i, DimSize, index);
            for (unsigned int j = 0; j < NDIM; j++) {
                index[j] = (DimSize[j] - index[j]) % DimSize[j];
            };
            ReversedIndex[i] = KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(DimSize, index);
        }
    }

    template<unsigned int NDIM>
    inline static unsigned int MortonZOrderFromRowMajorIndex(const unsigned int* DimSize, const unsigned int* Index)
    {
        unsigned int max_size = PowerOfTwo<Divide<MAX_MORTON_BITS, NDIM>::value>::value;
        //Since the output is limited by MAX_MORTON_BITS
        //indices with values larger than can be stored by
        //MAX_MORTON_BITS/NDIM will be truncated by the bitset constructor,
        //the largest index/dimension size value allowed is
        //2^{MAX_MORTON_BITS/NDIM}, for example with MAX_MORTON_BITS
        //set to 32, and NDIM=4, the max index allowed is 256
        for (unsigned int i = 0; i < NDIM; i++) {
            if (DimSize[i] >= max_size) {
                kfmout << "MortonZOrderFromRowMajorIndex: Error, ";
                kfmout << "dimension size " << DimSize[i] << " exceeds max ";
                kfmout << "allowable value of " << max_size << ".";
                kfmout << kfmendl;
                kfmexit(1);
            }
        }

        //interleaved bits from all the coordinates
        std::bitset<MAX_MORTON_BITS> interleaved_bits;
        //now compute the bits of each coordinate and insert them into the interleaved bits
        for (unsigned int i = 0; i < NDIM; i++) {
            std::bitset<Divide<MAX_MORTON_BITS, NDIM>::value> coord_bits(Index[i]);
            for (unsigned int j = 0; j < Divide<MAX_MORTON_BITS, NDIM>::value; j++) {
                interleaved_bits[j * NDIM + i] = coord_bits[j];
            }
        }

        //now convert the value of the interleaved bits to int
        //this cast should be safe since MAX_MORTON_BITS is limited to the size of int
        return static_cast<unsigned int>(interleaved_bits.to_ulong());
    }

    template<unsigned int NDIM>
    inline static unsigned int MortonZOrderFromOffset(unsigned int offset, const unsigned int* DimSize)
    {
        unsigned int index[NDIM];
        RowMajorIndexFromOffset<NDIM>(offset, DimSize, index);
        return MortonZOrderFromRowMajorIndex<NDIM>(DimSize, index);
    }
};


//specialization for base case of power of two
template<> struct KFMArrayMath::PowerOfTwo<0>
{
    enum
    {
        value = 1
    };
};

//specialization for base case of divide
template<int numerator> struct KFMArrayMath::Divide<numerator, 1>
{
    enum
    {
        value = numerator
    };
};


}  // namespace KEMField

#endif /* KFMArrayMath_H__ */

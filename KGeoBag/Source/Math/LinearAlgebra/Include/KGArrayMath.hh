#ifndef KGArrayMath_HH__
#define KGArrayMath_HH__

#include <cmath>
#include <cstddef>

namespace KGeoBag
{


/**
*
*@file KGArrayMath.hh
*@class KGArrayMath
*@brief collection of math functions used for array indexing
*@details
*
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue May 28 21:59:09 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KGArrayMath
{
  public:
    KGArrayMath() = default;
    ;
    virtual ~KGArrayMath() = default;
    ;

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
                Index[i] = KGArrayMath::Modulus(offset, DimSize[i]);
                div[i] = (offset - Index[i]) / DimSize[i];
            }
            else {
                Index[i] = KGArrayMath::Modulus(div[i + 1], DimSize[i]);
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
};


//specialization for base case
template<> struct KGArrayMath::PowerOfTwo<0>
{
    enum
    {
        value = 1
    };
};


}  // namespace KGeoBag

#endif /* KGArrayMath_H__ */

#ifndef KGArrayMath_HH__
#define KGArrayMath_HH__

#include <cmath>
#include <cstddef>

namespace KGeoBag{


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
        KGArrayMath(){};
        virtual ~KGArrayMath(){};

        //modulus of two integers
        static size_t Modulus(int arg, int n)
        {
            //returns arg mod n;
            double div = ( (double)arg )/( (double) n);
            return (size_t)(std::fabs( (double)arg - std::floor(div)*((double)n) ) );
        }

        //for a multidimensional array (using row major indexing) which has the
        //dimensions specified in DimSize, this function computes the offset from
        //the first element given the indices in the array Index
        template<size_t NDIM> inline static size_t
        OffsetFromRowMajorIndex(const size_t* DimSize, const size_t* Index)
        {
            size_t val = Index[0];
            for(size_t i=1; i<NDIM; i++)
            {
                val *= DimSize[i];
                val += Index[i];
            }
            return val;
        }

        //for a multidimensional array (using row major indexing) which has the
        //dimensions specified in DimSize, this function computes the stride between
        //consecutive elements in the selected dimension given that the other indices are fixed
        //the first element given the indices in the array Index
        template<size_t NDIM> inline static size_t
        StrideFromRowMajorIndex(size_t selected_dim, const size_t* DimSize)
        {
            size_t val = 1;
            for(size_t i=0; i<NDIM; i++)
            {
                if(i > selected_dim){val *= DimSize[i];};
            }
            return val;
        }



        //for a multidimensional array (using row major indexing) which has the
        //dimensions specified in DimSize, this function computes the indices of
        //the elements which has the given offset from the first element
        template<size_t NDIM> inline static void
        RowMajorIndexFromOffset(size_t offset, const size_t* DimSize, size_t* Index)
        {
            size_t div[NDIM];

            //in row major format the last index varies the fastest
            size_t i;
            for(size_t d=0; d < NDIM; d++)
            {
                i = NDIM - d -1;

                if(d == 0)
                {
                    Index[i] = KGArrayMath::Modulus(offset, DimSize[i]);
                    div[i] = (offset - Index[i])/DimSize[i];
                }
                else
                {
                    Index[i] = KGArrayMath::Modulus(div[i+1], DimSize[i]);
                    div[i] = (div[i+1] - Index[i])/DimSize[i];
                }
            }
        }

        //checks if all the indices in Index are in the valid range
        template<size_t NDIM> inline static bool
        CheckIndexValidity(const size_t* DimSize, const size_t* Index)
        {
            for(size_t i=0; i<NDIM; i++)
            {
                if(Index[i] >= DimSize[i]){return false;};
            }
            return true;
        };


        //given the dimensions of an array, computes its total size, assuming all dimensions are non-zero
        template<size_t NDIM> inline static size_t
        TotalArraySize(const size_t* DimSize)
        {
            size_t val = 1;
            for(size_t i=0; i<NDIM; i++)
            {
                val *= DimSize[i];
            }
            return val;
        }

        //compute 2^N at compile time
        template <size_t N>
        struct PowerOfTwo
        {
            enum { value = 2 * PowerOfTwo<N - 1>::value };
        };

};


//specialization for base case
template <>
struct KGArrayMath::PowerOfTwo<0>
{
    enum { value = 1 };
};



}//end of KEMField namespace

#endif /* KGArrayMath_H__ */

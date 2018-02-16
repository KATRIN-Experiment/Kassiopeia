#ifndef KFMPointwiseArrayReversedConjugateMultiplier_H__
#define KFMPointwiseArrayReversedConjugateMultiplier_H__

#include "KFMBinaryArrayOperator.hh"
#include "KFMArrayMath.hh"

#include <complex>

namespace KEMField{

/**
*
*@file KFMPointwiseArrayReversedConjugateMultiplier.hh
*@class KFMPointwiseArrayReversedConjugateMultiplier
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Sep 28 15:39:37 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>
class KFMPointwiseArrayReversedConjugateMultiplier: public KFMBinaryArrayOperator< std::complex<double>, NDIM>
{
    public:

        KFMPointwiseArrayReversedConjugateMultiplier()
        {
            fReversedIndexArray = NULL;
            fInitialized = false;
            for(unsigned int i=0; i<NDIM; i++){fDim[i] = 0;};
        };

        virtual ~KFMPointwiseArrayReversedConjugateMultiplier()
        {
            delete[] fReversedIndexArray;
        };

        virtual void Initialize()
        {
            if(this->fFirstInput != NULL)
            {
                for(unsigned int i=0; i<NDIM; i++)
                {
                    if( fDim[i] != this->fFirstInput->GetArrayDimension(i)){fInitialized = false;};
                };
            }

            if(!fInitialized && this->fFirstInput != NULL)
            {
                unsigned int n_elem  = this->fFirstInput->GetArraySize();
                this->fFirstInput->GetArrayDimensions(fDim);
                fReversedIndexArray = new unsigned int[n_elem];
                unsigned int ri;
                for(unsigned int i=0; i < n_elem; i++)
                {
                    KFMArrayMath::RowMajorIndexFromOffset<NDIM>(i, fDim, fIndex);
                    for(unsigned int j=0; j<NDIM; j++){fIndex[j] = (fDim[j] - fIndex[j])%fDim[j];};
                    ri = KFMArrayMath::OffsetFromRowMajorIndex<NDIM>(fDim, fIndex);
                    fReversedIndexArray[i] = ri;
                }
                fInitialized = true;
            }
        };

        virtual void GetReversedIndexArray(unsigned int* arr)
        {
            if(fInitialized)
            {
                unsigned int n_elem  = this->fFirstInput->GetArraySize();
                for(unsigned int i=0; i < n_elem; i++)
                {
                    arr[i] = fReversedIndexArray[i];
                }
            }
        }


        virtual void ExecuteOperation()
        {
            if(IsInputOutputValid())
            {
                std::complex<double>* in1ptr = this->fFirstInput->GetData();
                std::complex<double>* in2ptr = this->fSecondInput->GetData();
                std::complex<double>* outptr = this->fOutput->GetData();

                unsigned int n_elem = this->fFirstInput->GetArraySize();


                this->fSecondInput->GetArrayDimensions(fDim);

                for(unsigned int i=0; i < n_elem; i++)
                {
                    outptr[i] = (in1ptr[i])*( std::conj( in2ptr[ fReversedIndexArray[i] ] ) );
                }
            }
        }

    private:

        unsigned int fIndex[NDIM];
        unsigned int fDim[NDIM];

        unsigned int* fReversedIndexArray;
        bool fInitialized;


        virtual bool IsInputOutputValid() const
        {
            if(this->fFirstInput && this->fSecondInput && this->fOutput )
            {
                //check they have the same size/num elements
                if( this->HaveSameNumberOfElements(this->fFirstInput, this->fOutput) &&
                    this->HaveSameNumberOfElements(this->fSecondInput, this->fOutput)  )
                {
                    //check they have the same dimensions/shape
                    if(this->HaveSameDimensions(this->fFirstInput, this->fOutput) &&
                       this->HaveSameDimensions(this->fSecondInput, this->fOutput)  )
                    {
                        return true;
                    }
                }
            }
            return false;
        }


};


}

#endif /* __KFMPointwiseArrayReversedConjugateMultiplier_H__ */

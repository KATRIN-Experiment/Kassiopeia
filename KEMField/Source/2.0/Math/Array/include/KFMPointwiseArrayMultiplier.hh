#ifndef KFMPointwiseArrayMultiplier_H__
#define KFMPointwiseArrayMultiplier_H__

#include "KFMBinaryArrayOperator.hh"

namespace KEMField{

/**
*
*@file KFMPointwiseArrayMultiplier.hh
*@class KFMPointwiseArrayMultiplier
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Sep 28 15:39:37 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ArrayType, unsigned int NDIM>
class KFMPointwiseArrayMultiplier: public KFMBinaryArrayOperator< ArrayType, NDIM>
{
    public:
        KFMPointwiseArrayMultiplier(){};
        virtual ~KFMPointwiseArrayMultiplier(){};

        virtual void Initialize(){;};

        virtual void ExecuteOperation()
        {
            if(IsInputOutputValid())
            {
                ArrayType* in1ptr = this->fFirstInput->GetData();
                ArrayType* in2ptr = this->fSecondInput->GetData();
                ArrayType* outptr = this->fOutput->GetData();

                unsigned int n_elem = this->fFirstInput->GetArraySize();

                for(unsigned int i=0; i < n_elem; i++)
                {
                    outptr[i] = (in1ptr[i])*(in2ptr[i]);
                }
            }
        }

    private:

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

#endif /* __KFMPointwiseArrayMultiplier_H__ */

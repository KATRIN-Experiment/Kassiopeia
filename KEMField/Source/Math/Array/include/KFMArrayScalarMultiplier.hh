#ifndef KFM3DArrayScalarMultiplier_H__
#define KFM3DArrayScalarMultiplier_H__

#include "KFMUnaryArrayOperator.hh"

namespace KEMField
{

/**
*
*@file KFM3DArrayScalarMultiplier.hh
*@class KFM3DArrayScalarMultiplier
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Oct  3 10:42:16 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ArrayType, unsigned int NDIM>
class KFMArrayScalarMultiplier : public KFMUnaryArrayOperator<ArrayType, NDIM>
{
  public:
    KFMArrayScalarMultiplier() = default;
    ;
    ~KFMArrayScalarMultiplier() override = default;
    ;

    void SetScalarMultiplicationFactor(const ArrayType& fac)
    {
        fScalarFactor = fac;
    }

    void Initialize() override
    {
        ;
    };

    void ExecuteOperation() override
    {

        if (this->fInput != nullptr && this->fOutput != nullptr) {
            if (KFMArrayOperator<ArrayType, NDIM>::HaveSameNumberOfElements(this->fInput, this->fOutput)) {
                ArrayType* inptr = this->fInput->GetData();
                ArrayType* outptr = this->fOutput->GetData();

                unsigned int n_elem = this->fInput->GetArraySize();
                for (unsigned int i = 0; i < n_elem; i++) {
                    outptr[i] = (inptr[i]) * (fScalarFactor);
                }
            }
        }
    }


  protected:
    ArrayType fScalarFactor;
};


}  // namespace KEMField

#endif /* __KFM3DArrayScalarMultiplier_H__ */

#ifndef KPreconditioner_HH__
#define KPreconditioner_HH__

#include "KSquareMatrix.hh"

#include <string>

namespace KEMField
{


/*
*
*@file KPreconditioner.hh
*@class KPreconditioner
*@brief
* Stub class wrapping a square matrix that is used for preconditioning.
* The preconditioner must indicate if it is a stationary method:
* (does not vary from iteration to iteration), or if it is non-stationary
*
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Jun  3 09:42:09 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType> class KPreconditioner : public KSquareMatrix<ValueType>
{
  public:
    KPreconditioner() : KSquareMatrix<ValueType>(){};
    ~KPreconditioner() override = default;

    virtual std::string Name() = 0;

    virtual bool IsStationary() = 0;
};


}  // namespace KEMField

#endif /* KPreconditioner_H__ */

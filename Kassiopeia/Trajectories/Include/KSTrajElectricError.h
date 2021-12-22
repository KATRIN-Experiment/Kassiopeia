#ifndef Kassiopeia_KSTrajElectricError_h_
#define Kassiopeia_KSTrajElectricError_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSTrajElectricError : public KSMathArray<5>
{
  public:
    KSTrajElectricError();
    KSTrajElectricError(const KSTrajElectricError& anOperand);
    ~KSTrajElectricError() override;

    //**********
    //assignment
    //**********

  public:
    KSTrajElectricError& operator=(const double& anOperand);

    KSTrajElectricError& operator=(const KSMathArray<5>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajElectricError& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajElectricError& operator=(const KSTrajElectricError& anOperand);

    //*********
    //variables
    //*********

  public:
    const double& GetTimeError() const;
    const double& GetLengthError() const;
    const katrin::KThreeVector& GetPositionError() const;

  protected:
    mutable double fTimeError;
    mutable double fLengthError;
    mutable katrin::KThreeVector fPositionError;
};

inline KSTrajElectricError& KSTrajElectricError::operator=(const double& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajElectricError& KSTrajElectricError::operator=(const KSMathArray<5>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajElectricError& KSTrajElectricError::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajElectricError& KSTrajElectricError::operator=(const KSTrajElectricError& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

}  // namespace Kassiopeia

#endif

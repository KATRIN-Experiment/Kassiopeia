#ifndef Kassiopeia_KSTrajMagneticError_h_
#define Kassiopeia_KSTrajMagneticError_h_

#include "KSMathArray.h"
#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSTrajMagneticError : public KSMathArray<5>
{
  public:
    KSTrajMagneticError();
    KSTrajMagneticError(const KSTrajMagneticError& anOperand);
    ~KSTrajMagneticError() override;

    //**********
    //assignment
    //**********

  public:
    KSTrajMagneticError& operator=(const double& anOperand);

    KSTrajMagneticError& operator=(const KSMathArray<5>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajMagneticError& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajMagneticError& operator=(const KSTrajMagneticError& anOperand);

    //*********
    //variables
    //*********

  public:
    const double& GetTimeError() const;
    const double& GetLengthError() const;
    const KGeoBag::KThreeVector& GetPositionError() const;

  protected:
    mutable double fTimeError;
    mutable double fLengthError;
    mutable KGeoBag::KThreeVector fPositionError;
};

inline KSTrajMagneticError& KSTrajMagneticError::operator=(const double& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajMagneticError& KSTrajMagneticError::operator=(const KSMathArray<5>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajMagneticError& KSTrajMagneticError::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajMagneticError& KSTrajMagneticError::operator=(const KSTrajMagneticError& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

}  // namespace Kassiopeia

#endif

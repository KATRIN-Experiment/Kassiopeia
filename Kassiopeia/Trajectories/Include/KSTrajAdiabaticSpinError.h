#ifndef Kassiopeia_KSTrajAdiabaticSpinError_h_
#define Kassiopeia_KSTrajAdiabaticSpinError_h_

#include "KSMathArray.h"
#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSTrajAdiabaticSpinError : public KSMathArray<10>
{
  public:
    KSTrajAdiabaticSpinError();
    KSTrajAdiabaticSpinError(const KSTrajAdiabaticSpinError& anOperand);
    ~KSTrajAdiabaticSpinError() override;

    //**********
    //assignment
    //**********

  public:
    KSTrajAdiabaticSpinError& operator=(const double& anOperand);

    KSTrajAdiabaticSpinError& operator=(const KSMathArray<10>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajAdiabaticSpinError& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajAdiabaticSpinError& operator=(const KSTrajAdiabaticSpinError& anOperand);

    //*********
    //variables
    //*********

  public:
    const double& GetTimeError() const;
    const double& GetLengthError() const;
    const KGeoBag::KThreeVector& GetPositionError() const;
    const KGeoBag::KThreeVector& GetMomentumError() const;
    const double& GetAlignedSpinError() const;
    const double& GetSpinAngleError() const;

  protected:
    mutable double fTimeError;
    mutable double fLengthError;
    mutable KGeoBag::KThreeVector fPositionError;
    mutable KGeoBag::KThreeVector fMomentumError;
    mutable double fAlignedSpinError;
    mutable double fSpinAngleError;
};

inline KSTrajAdiabaticSpinError& KSTrajAdiabaticSpinError::operator=(const double& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

inline KSTrajAdiabaticSpinError& KSTrajAdiabaticSpinError::operator=(const KSMathArray<10>& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajAdiabaticSpinError&
KSTrajAdiabaticSpinError::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

inline KSTrajAdiabaticSpinError& KSTrajAdiabaticSpinError::operator=(const KSTrajAdiabaticSpinError& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

}  // namespace Kassiopeia

#endif

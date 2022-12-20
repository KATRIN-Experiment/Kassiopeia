#ifndef Kassiopeia_KSTrajExactTrappedError_h_
#define Kassiopeia_KSTrajExactTrappedError_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSTrajExactTrappedError : public KSMathArray<8>
{
  public:
    KSTrajExactTrappedError();
    KSTrajExactTrappedError(const KSTrajExactTrappedError& anOperand);
    ~KSTrajExactTrappedError() override;

    //**********
    //assignment
    //**********

  public:
    KSTrajExactTrappedError& operator=(const double& anOperand);

    KSTrajExactTrappedError& operator=(const KSMathArray<8>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajExactTrappedError& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajExactTrappedError& operator=(const KSTrajExactTrappedError& anOperand);

    //*********
    //variables
    //*********

  public:
    const double& GetTimeError() const;
    const double& GetLengthError() const;
    const katrin::KThreeVector& GetPositionError() const;
    const katrin::KThreeVector& GetMomentumError() const;

  protected:
    mutable double fTimeError;
    mutable double fLengthError;
    mutable katrin::KThreeVector fPositionError;
    mutable katrin::KThreeVector fMomentumError;
};

inline KSTrajExactTrappedError& KSTrajExactTrappedError::operator=(const double& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    return *this;
}

inline KSTrajExactTrappedError& KSTrajExactTrappedError::operator=(const KSMathArray<8>& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajExactTrappedError&
KSTrajExactTrappedError::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    return *this;
}

inline KSTrajExactTrappedError& KSTrajExactTrappedError::operator=(const KSTrajExactTrappedError& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    return *this;
}

}  // namespace Kassiopeia

#endif

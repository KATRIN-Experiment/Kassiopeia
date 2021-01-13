#ifndef Kassiopeia_KSTrajMagneticDerivative_h_
#define Kassiopeia_KSTrajMagneticDerivative_h_

#include "KSMathArray.h"
#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSTrajMagneticDerivative : public KSMathArray<5>
{
  public:
    KSTrajMagneticDerivative();
    KSTrajMagneticDerivative(const KSTrajMagneticDerivative& anOperand);
    ~KSTrajMagneticDerivative() override;

    //**********
    //assignment
    //**********

  public:
    KSTrajMagneticDerivative& operator=(const double& anOperand);

    KSTrajMagneticDerivative& operator=(const KSMathArray<5>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajMagneticDerivative& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajMagneticDerivative& operator=(const KSTrajMagneticDerivative& anOperand);

    //*********
    //variables
    //*********

  public:
    void AddToTime(const double& aTime);
    void AddToSpeed(const double& aSpeed);
    void AddToVelocity(const KGeoBag::KThreeVector& aVelocity);
};

inline KSTrajMagneticDerivative& KSTrajMagneticDerivative::operator=(const double& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajMagneticDerivative& KSTrajMagneticDerivative::operator=(const KSMathArray<5>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajMagneticDerivative&
KSTrajMagneticDerivative::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajMagneticDerivative& KSTrajMagneticDerivative::operator=(const KSTrajMagneticDerivative& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

}  // namespace Kassiopeia

#endif

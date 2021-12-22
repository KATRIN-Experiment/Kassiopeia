#ifndef Kassiopeia_KSTrajElectricDerivative_h_
#define Kassiopeia_KSTrajElectricDerivative_h_

#include "KSMathArray.h"

#include "KThreeVector.hh"

namespace Kassiopeia
{

class KSTrajElectricDerivative : public KSMathArray<5>
{
  public:
    KSTrajElectricDerivative();
    KSTrajElectricDerivative(const KSTrajElectricDerivative& anOperand);
    ~KSTrajElectricDerivative() override;

    //**********
    //assignment
    //**********

  public:
    KSTrajElectricDerivative& operator=(const double& anOperand);

    KSTrajElectricDerivative& operator=(const KSMathArray<5>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajElectricDerivative& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajElectricDerivative& operator=(const KSTrajElectricDerivative& anOperand);

    //*********
    //variables
    //*********

  public:
    void AddToTime(const double& aTime);
    void AddToSpeed(const double& aSpeed);
    void AddToVelocity(const katrin::KThreeVector& aVelocity);
};

inline KSTrajElectricDerivative& KSTrajElectricDerivative::operator=(const double& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajElectricDerivative& KSTrajElectricDerivative::operator=(const KSMathArray<5>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajElectricDerivative&
KSTrajElectricDerivative::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

inline KSTrajElectricDerivative& KSTrajElectricDerivative::operator=(const KSTrajElectricDerivative& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    return *this;
}

}  // namespace Kassiopeia

#endif

#ifndef Kassiopeia_KSTrajAdiabaticSpinDerivative_h_
#define Kassiopeia_KSTrajAdiabaticSpinDerivative_h_

#include "KSMathArray.h"
#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{

class KSTrajAdiabaticSpinDerivative : public KSMathArray<10>
{
  public:
    KSTrajAdiabaticSpinDerivative();
    KSTrajAdiabaticSpinDerivative(const KSTrajAdiabaticSpinDerivative& anOperand);
    virtual ~KSTrajAdiabaticSpinDerivative();

    //**********
    //assignment
    //**********

  public:
    KSTrajAdiabaticSpinDerivative& operator=(const double& anOperand);

    KSTrajAdiabaticSpinDerivative& operator=(const KSMathArray<10>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajAdiabaticSpinDerivative& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajAdiabaticSpinDerivative& operator=(const KSTrajAdiabaticSpinDerivative& anOperand);

    //*********
    //variables
    //*********

  public:
    void AddToTime(const double& aTime);
    void AddToSpeed(const double& aSpeed);
    void AddToVelocity(const KThreeVector& aVelocity);
    void AddToForce(const KThreeVector& aForce);
    void AddToMDot(const double& anMDot);
    void AddToPhiDot(const double& aPhiDot);
};

inline KSTrajAdiabaticSpinDerivative& KSTrajAdiabaticSpinDerivative::operator=(const double& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

inline KSTrajAdiabaticSpinDerivative& KSTrajAdiabaticSpinDerivative::operator=(const KSMathArray<10>& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajAdiabaticSpinDerivative&
KSTrajAdiabaticSpinDerivative::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

inline KSTrajAdiabaticSpinDerivative&
KSTrajAdiabaticSpinDerivative::operator=(const KSTrajAdiabaticSpinDerivative& anOperand)
{
    this->KSMathArray<10>::operator=(anOperand);
    return *this;
}

}  // namespace Kassiopeia

#endif

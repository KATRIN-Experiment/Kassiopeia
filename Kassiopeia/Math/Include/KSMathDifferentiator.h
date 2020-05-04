#ifndef Kassiopeia_KSMathDifferentiator_h_
#define Kassiopeia_KSMathDifferentiator_h_

#include "KSMathSystem.h"

namespace Kassiopeia
{

template<class XSystemType> class KSMathDifferentiator;

template<class XValueType, class XDerivativeType, class XErrorType>
class KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>
{
  public:
    KSMathDifferentiator();
    virtual ~KSMathDifferentiator();

  public:
    virtual void Differentiate(double aTime, const XValueType& aValue, XDerivativeType& aDerivative) const = 0;
};

template<class XValueType, class XDerivativeType, class XErrorType>
KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>::KSMathDifferentiator()
{}

template<class XValueType, class XDerivativeType, class XErrorType>
KSMathDifferentiator<KSMathSystem<XValueType, XDerivativeType, XErrorType>>::~KSMathDifferentiator()
{}

}  // namespace Kassiopeia

#endif

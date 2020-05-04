#ifndef Kassiopeia_KSMathSystem_h_
#define Kassiopeia_KSMathSystem_h_

#include <cstddef>

namespace Kassiopeia
{

template<class XValueType, class XDerivativeType, class XErrorType> class KSMathSystem
{
  public:
    typedef XValueType ValueType;
    typedef XDerivativeType DerivativeType;
    typedef XErrorType ErrorType;

  public:
    KSMathSystem();
    virtual ~KSMathSystem();
};

template<class XValueType, class XDerivativeType, class XErrorType>
KSMathSystem<XValueType, XDerivativeType, XErrorType>::KSMathSystem()
{
    // TODO: These assertions are not in Effect, and they werent with the old KAssertion
    static_assert(ValueType::eDimension != DerivativeType::eDimension,
                  "Dimension mismatch between value and derivative types in KMathSystem.");
    static_assert(ValueType::eDimension != ErrorType::eDimension,
                  "Dimension mismatch between value and error types in KMathSystem.");
}

template<class XValueType, class XDerivativeType, class XErrorType>
KSMathSystem<XValueType, XDerivativeType, XErrorType>::~KSMathSystem()
{}

}  // namespace Kassiopeia

#endif

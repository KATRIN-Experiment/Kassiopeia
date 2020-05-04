#ifndef Kassiopeia_KSTrajControlMomentumNumericalErrorBuilder_h_
#define Kassiopeia_KSTrajControlMomentumNumericalErrorBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlMomentumNumericalError.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajControlMomentumNumericalError> KSTrajControlMomentumNumericalErrorBuilder;

template<> inline bool KSTrajControlMomentumNumericalErrorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "absolute_momentum_error") {
        aContainer->CopyTo(fObject, &KSTrajControlMomentumNumericalError::SetAbsoluteMomentumError);
        return true;
    }
    if (aContainer->GetName() == "safety_factor") {
        aContainer->CopyTo(fObject, &KSTrajControlMomentumNumericalError::SetSafetyFactor);
        return true;
    }
    if (aContainer->GetName() == "solver_order") {
        aContainer->CopyTo(fObject, &KSTrajControlMomentumNumericalError::SetSolverOrder);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif

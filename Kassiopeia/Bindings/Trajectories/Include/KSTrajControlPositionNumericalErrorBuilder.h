#ifndef Kassiopeia_KSTrajControlPositionNumericalErrorBuilder_h_
#define Kassiopeia_KSTrajControlPositionNumericalErrorBuilder_h_

#include "KComplexElement.hh"
#include "KSTrajControlPositionNumericalError.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTrajControlPositionNumericalError> KSTrajControlPositionNumericalErrorBuilder;

template<> inline bool KSTrajControlPositionNumericalErrorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "absolute_position_error") {
        aContainer->CopyTo(fObject, &KSTrajControlPositionNumericalError::SetAbsolutePositionError);
        return true;
    }
    if (aContainer->GetName() == "safety_factor") {
        aContainer->CopyTo(fObject, &KSTrajControlPositionNumericalError::SetSafetyFactor);
        return true;
    }
    if (aContainer->GetName() == "solver_order") {
        aContainer->CopyTo(fObject, &KSTrajControlPositionNumericalError::SetSolverOrder);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif

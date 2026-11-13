#ifndef Kassiopeia_KSIntCalculatorConstantBuilder_h_
#define Kassiopeia_KSIntCalculatorConstantBuilder_h_

#include "KComplexElement.hh"
#include "KSIntCalculatorConstant.h"

using namespace Kassiopeia;
namespace katrin
{

//! this calculator returns a constant cross section and scatters isotropically.
//! @example <ksint_calculator_constant name="calculator_constant" cross_section="1.e-18"/>
typedef KComplexElement<KSIntCalculatorConstant> KSIntCalculatorConstantBuilder;

template<> inline bool KSIntCalculatorConstantBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "cross_section") {
        aContainer->CopyTo(fObject, &KSIntCalculatorConstant::SetCrossSection);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

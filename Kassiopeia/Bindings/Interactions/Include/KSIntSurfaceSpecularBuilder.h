#ifndef Kassiopeia_KSIntSurfaceSpecularBuilder_h_
#define Kassiopeia_KSIntSurfaceSpecularBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSurfaceSpecular.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSIntSurfaceSpecular> KSIntSurfaceSpecularBuilder;

template<> inline bool KSIntSurfaceSpecularBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "probability") {
        aContainer->CopyTo(fObject, &KSIntSurfaceSpecular::SetProbability);
        return true;
    }
    if (aContainer->GetName() == "reflection_loss") {
        aContainer->CopyTo(fObject, &KSIntSurfaceSpecular::SetReflectionLoss);
        return true;
    }
    if (aContainer->GetName() == "transmission_loss") {
        aContainer->CopyTo(fObject, &KSIntSurfaceSpecular::SetTransmissionLoss);
        return true;
    }
    if (aContainer->GetName() == "reflection_loss_fraction") {
        aContainer->CopyTo(fObject, &KSIntSurfaceSpecular::SetReflectionLossFraction);
        return true;
    }
    if (aContainer->GetName() == "transmission_loss_fraction") {
        aContainer->CopyTo(fObject, &KSIntSurfaceSpecular::SetTransmissionLossFraction);
        return true;
    }
    return false;
}

}  // namespace katrin
#endif

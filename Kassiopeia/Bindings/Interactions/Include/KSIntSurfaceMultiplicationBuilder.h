#ifndef Kassiopeia_KSIntSurfaceMultiplicationBuilder_h_
#define Kassiopeia_KSIntSurfaceMultiplicationBuilder_h_

#include "KComplexElement.hh"
#include "KSIntSurfaceMultiplication.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSIntSurfaceMultiplication> KSIntSurfaceMultiplicationBuilder;

template<> inline bool KSIntSurfaceMultiplicationBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "side") {
        aContainer->CopyTo(fObject, &KSIntSurfaceMultiplication::SetSide);
        return true;
    }
    if (aContainer->GetName() == "energy_loss_fraction") {
        aContainer->CopyTo(fObject, &KSIntSurfaceMultiplication::SetEnergyLossFraction);
        return true;
    }
    if (aContainer->GetName() == "required_energy_per_particle_ev") {
        aContainer->CopyTo(fObject, &KSIntSurfaceMultiplication::SetEnergyRequiredPerParticle);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

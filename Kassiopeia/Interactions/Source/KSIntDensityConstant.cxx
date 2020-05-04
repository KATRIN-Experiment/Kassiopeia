#include "KSIntDensityConstant.h"

#include "KConst.h"

namespace Kassiopeia
{

KSIntDensityConstant::KSIntDensityConstant() : fTemperature(0.), fPressure(0.), fDensity(-1.) {}
KSIntDensityConstant::KSIntDensityConstant(const KSIntDensityConstant& aCopy) :
    KSComponent(),
    fTemperature(aCopy.fTemperature),
    fPressure(aCopy.fPressure),
    fDensity(aCopy.fDensity)
{}
KSIntDensityConstant* KSIntDensityConstant::Clone() const
{
    return new KSIntDensityConstant(*this);
}
KSIntDensityConstant::~KSIntDensityConstant() {}

void KSIntDensityConstant::CalculateDensity(const KSParticle&, double& aDensity)
{
    if (fDensity == -1.) {
        aDensity = fPressure / (katrin::KConst::kB() * fTemperature);
        return;
    }
    else {
        aDensity = fDensity;
        return;
    }
}

}  // namespace Kassiopeia

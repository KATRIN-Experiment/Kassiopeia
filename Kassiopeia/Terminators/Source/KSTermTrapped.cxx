#include "KSTermTrapped.h"

#include "KSTerminatorsMessage.h"

namespace Kassiopeia
{

KSTermTrapped::KSTermTrapped() :
    fUseMagneticField(true),
    fUseElectricField(false),
    fMaxTurns(1),
    fCurrentTurns(0),
    fCurrentDotProduct(0)
{}
KSTermTrapped::KSTermTrapped(const KSTermTrapped& aCopy) :
    KSComponent(aCopy),
    fUseMagneticField(aCopy.fUseMagneticField),
    fUseElectricField(aCopy.fUseElectricField),
    fMaxTurns(aCopy.fMaxTurns),
    fCurrentTurns(aCopy.fCurrentTurns),
    fCurrentDotProduct(aCopy.fCurrentDotProduct)
{}
KSTermTrapped* KSTermTrapped::Clone() const
{
    return new KSTermTrapped(*this);
}
KSTermTrapped::~KSTermTrapped() = default;

void KSTermTrapped::SetUseMagneticField(bool aFlag)
{
    fUseMagneticField = aFlag;
    if (fUseMagneticField)
        fUseElectricField = false;
}
void KSTermTrapped::SetUseElectricField(bool aFlag)
{
    fUseElectricField = aFlag;
    if (fUseElectricField)
        fUseMagneticField = false;
}

void KSTermTrapped::ActivateComponent()
{
    if (fUseElectricField && fUseMagneticField)
        termmsg(eError) << "terminator <" << GetName() << " can use either electric or magnetic field, but not both" << eom;
    else if (!fUseElectricField && !fUseMagneticField)
        termmsg(eError) << "terminator <" << GetName() << " must use either electric or magnetic field" << eom;

    fCurrentTurns = 0;
    fCurrentDotProduct = 0;
    return;
}
void KSTermTrapped::DeactivateComponent()
{
    fCurrentTurns = 0;
    fCurrentDotProduct = 0;
    return;
}

void KSTermTrapped::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (fMaxTurns < 0)
        termmsg(eError) << "negative maximal number of turns defined in Trapped terminator <" << GetName() << ">"
                        << eom;


    double DotProduct = 0.;
    if (fUseMagneticField)
        DotProduct = anInitialParticle.GetMagneticField().Dot(anInitialParticle.GetMomentum());
    else if (fUseElectricField)
        DotProduct = anInitialParticle.GetElectricField().Dot(anInitialParticle.GetMomentum());

    if (DotProduct * fCurrentDotProduct < 0.) {
        fCurrentTurns += 1;
        fCurrentDotProduct = DotProduct;
        if (fCurrentTurns >= fMaxTurns) {
            fCurrentTurns = 0;
            fCurrentDotProduct = 0;

            aFlag = true;
            return;
        }
    }

    fCurrentDotProduct = DotProduct;
    aFlag = false;
    return;
}
void KSTermTrapped::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia

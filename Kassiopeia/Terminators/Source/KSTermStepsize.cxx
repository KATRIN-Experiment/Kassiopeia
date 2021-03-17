#include "KSTermStepsize.h"

#include "KSTerminatorsMessage.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

KSTermStepsize::KSTermStepsize() :
    fLowerLimit(numeric_limits<double>::min()),
    fUpperLimit(numeric_limits<double>::max()),
    fCurrentPathLength(0.)
{}
KSTermStepsize::KSTermStepsize(const KSTermStepsize& aCopy) :
    KSComponent(aCopy),
    fLowerLimit(aCopy.fLowerLimit),
    fUpperLimit(aCopy.fUpperLimit),
    fCurrentPathLength(0.)
{}
KSTermStepsize* KSTermStepsize::Clone() const
{
    return new KSTermStepsize(*this);
}
KSTermStepsize::~KSTermStepsize() = default;

void KSTermStepsize::CalculateTermination(const KSParticle& anInitialParticle, bool& aFlag)
{
    if (fLowerLimit < 0. || fUpperLimit < 0.)
        termmsg(eError) << "negative stepsize limit defined in Stepsize terminator <" << GetName() << ">" << eom;

    if (fCurrentPathLength > 0.) {
        double tStepsize = fabs(anInitialParticle.GetLength() - fCurrentPathLength);
        if (tStepsize < fLowerLimit || tStepsize > fUpperLimit) {
            aFlag = true;
            return;
        }
    }

    fCurrentPathLength = anInitialParticle.GetLength();

    aFlag = false;
    return;
}
void KSTermStepsize::ExecuteTermination(const KSParticle&, KSParticle& aFinalParticle, KSParticleQueue&) const
{
    aFinalParticle.SetActive(false);
    aFinalParticle.SetLabel(GetName());
    return;
}

}  // namespace Kassiopeia

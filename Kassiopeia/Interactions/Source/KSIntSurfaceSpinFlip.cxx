#include "KSIntSurfaceSpinFlip.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

#include "KConst.h"

#include <cmath>
#include <iostream>

namespace Kassiopeia
{

KSIntSurfaceSpinFlip::KSIntSurfaceSpinFlip() : fProbability(.0)
{
    std::cout << "/* spin flip interaction created */" << std::endl;
}
KSIntSurfaceSpinFlip::KSIntSurfaceSpinFlip(const KSIntSurfaceSpinFlip& aCopy) :
    KSComponent(),
    fProbability(aCopy.fProbability)
{
    std::cout << "/* spin flip interaction created */" << std::endl;
}
KSIntSurfaceSpinFlip* KSIntSurfaceSpinFlip::Clone() const
{
    return new KSIntSurfaceSpinFlip(*this);
}
KSIntSurfaceSpinFlip::~KSIntSurfaceSpinFlip() {}

void KSIntSurfaceSpinFlip::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                              KSParticleQueue& /*aQueue*/)
{
    std::cout << "/* spin flip interaction executed */" << std::endl;
    double tChoice = KRandom::GetInstance().Uniform(0., 1.);
    if (tChoice < fProbability) {
        aFinalParticle = anInitialParticle;
        aFinalParticle.SetAlignedSpin(-1.0 * anInitialParticle.GetAlignedSpin());
        aFinalParticle.SetSpinAngle(-1.0 * anInitialParticle.GetSpinAngle());
    }
    return;
}

}  // namespace Kassiopeia

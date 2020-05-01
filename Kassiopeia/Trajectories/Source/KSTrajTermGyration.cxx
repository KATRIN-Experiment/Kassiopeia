#include "KSTrajTermGyration.h"

#include "KConst.h"
#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

KSTrajTermGyration::KSTrajTermGyration() {}
KSTrajTermGyration::KSTrajTermGyration(const KSTrajTermGyration&) : KSComponent() {}
KSTrajTermGyration* KSTrajTermGyration::Clone() const
{
    return new KSTrajTermGyration(*this);
}
KSTrajTermGyration::~KSTrajTermGyration() {}

void KSTrajTermGyration::Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle,
                                       KSTrajAdiabaticDerivative& aDerivative) const
{
    double tPhaseVelocity = 2. * katrin::KConst::Pi() * aParticle.GetCyclotronFrequency();

    aDerivative.AddToPhaseVelocity(tPhaseVelocity);

    fPhaseVelocity = tPhaseVelocity;
    return;
}

STATICINT sKSTrajTermGyrationDict =
    KSDictionary<KSTrajTermGyration>::AddComponent(&KSTrajTermGyration::GetPhaseVelocity, "phase_velocity");
}  // namespace Kassiopeia

#include "KSTrajControlSpinPrecession.h"

#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

KSTrajControlSpinPrecession::KSTrajControlSpinPrecession() : fFraction(1. / 16.) {}
KSTrajControlSpinPrecession::KSTrajControlSpinPrecession(const KSTrajControlSpinPrecession& aCopy) :
    KSComponent(aCopy),
    fFraction(aCopy.fFraction)
{}
KSTrajControlSpinPrecession* KSTrajControlSpinPrecession::Clone() const
{
    return new KSTrajControlSpinPrecession(*this);
}
KSTrajControlSpinPrecession::~KSTrajControlSpinPrecession() = default;

void KSTrajControlSpinPrecession::Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue)
{
    double tSpinPrecessionFrequency = aParticle.GetSpinPrecessionFrequency();
    aValue = fFraction / tSpinPrecessionFrequency;
    return;
}
void KSTrajControlSpinPrecession::Check(const KSTrajExactSpinParticle&, const KSTrajExactSpinParticle&,
                                        const KSTrajExactSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlSpinPrecession::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    double tSpinPrecessionFrequency = aParticle.GetSpinPrecessionFrequency();
    aValue = fFraction / tSpinPrecessionFrequency;
    return;
}
void KSTrajControlSpinPrecession::Check(const KSTrajAdiabaticSpinParticle&, const KSTrajAdiabaticSpinParticle&,
                                        const KSTrajAdiabaticSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}


}  // namespace Kassiopeia

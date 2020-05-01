#include "KSTrajControlCyclotron.h"

#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

KSTrajControlCyclotron::KSTrajControlCyclotron() : fFraction(1. / 16.) {}
KSTrajControlCyclotron::KSTrajControlCyclotron(const KSTrajControlCyclotron& aCopy) :
    KSComponent(),
    fFraction(aCopy.fFraction)
{}
KSTrajControlCyclotron* KSTrajControlCyclotron::Clone() const
{
    return new KSTrajControlCyclotron(*this);
}
KSTrajControlCyclotron::~KSTrajControlCyclotron() {}

void KSTrajControlCyclotron::Calculate(const KSTrajExactParticle& aParticle, double& aValue)
{
    double tCyclotronFrequency = aParticle.GetCyclotronFrequency();
    aValue = fFraction / tCyclotronFrequency;
    return;
}
void KSTrajControlCyclotron::Check(const KSTrajExactParticle&, const KSTrajExactParticle&, const KSTrajExactError&,
                                   bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlCyclotron::Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue)
{
    double tCyclotronFrequency = aParticle.GetCyclotronFrequency();
    aValue = fFraction / tCyclotronFrequency;
    return;
}
void KSTrajControlCyclotron::Check(const KSTrajExactSpinParticle&, const KSTrajExactSpinParticle&,
                                   const KSTrajExactSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlCyclotron::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    double tCyclotronFrequency = aParticle.GetCyclotronFrequency();
    aValue = fFraction / tCyclotronFrequency;
    return;
}
void KSTrajControlCyclotron::Check(const KSTrajAdiabaticSpinParticle&, const KSTrajAdiabaticSpinParticle&,
                                   const KSTrajAdiabaticSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlCyclotron::Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue)
{
    double tCyclotronFrequency = aParticle.GetCyclotronFrequency();
    aValue = fFraction / tCyclotronFrequency;
    return;
}
void KSTrajControlCyclotron::Check(const KSTrajAdiabaticParticle&, const KSTrajAdiabaticParticle&,
                                   const KSTrajAdiabaticError&, bool& aFlag)
{
    aFlag = true;
    return;
}

}  // namespace Kassiopeia

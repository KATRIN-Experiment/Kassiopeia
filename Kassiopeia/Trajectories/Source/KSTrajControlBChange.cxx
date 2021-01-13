#include "KSTrajControlBChange.h"

#include "KSTrajectoriesMessage.h"

#include <cmath>

namespace Kassiopeia
{

KSTrajControlBChange::KSTrajControlBChange() : fFraction(1. / 16.) {}
KSTrajControlBChange::KSTrajControlBChange(const KSTrajControlBChange& aCopy) :
    KSComponent(aCopy),
    fFraction(aCopy.fFraction)
{}
KSTrajControlBChange* KSTrajControlBChange::Clone() const
{
    return new KSTrajControlBChange(*this);
}
KSTrajControlBChange::~KSTrajControlBChange() = default;

void KSTrajControlBChange::Calculate(const KSTrajExactParticle& aParticle, double& aValue)
{
    double MaxGradient = 0.;
    for (int i = 0; i < 9; i++) {
        if (std::abs((aParticle.GetMagneticField())[i]) > MaxGradient) {
            MaxGradient = std::abs((aParticle.GetMagneticField())[i]);
        }
    }
    aValue = aParticle.GetMagneticField().Magnitude() / MaxGradient * fFraction;
    return;
}
void KSTrajControlBChange::Check(const KSTrajExactParticle&, const KSTrajExactParticle&, const KSTrajExactError&,
                                 bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlBChange::Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue)
{
    double MaxGradient = 0.;
    for (int i = 0; i < 9; i++) {
        if (std::abs((aParticle.GetMagneticField())[i]) > MaxGradient) {
            MaxGradient = std::abs((aParticle.GetMagneticField())[i]);
        }
    }
    aValue = aParticle.GetMagneticField().Magnitude() / MaxGradient * fFraction;
    return;
}
void KSTrajControlBChange::Check(const KSTrajExactSpinParticle&, const KSTrajExactSpinParticle&,
                                 const KSTrajExactSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlBChange::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    // std::cout << "MOO\n";
    double MaxGradient = 0.;
    for (int i = 0; i < 9; i++) {
        if (std::abs((aParticle.GetMagneticField())[i]) > MaxGradient) {
            MaxGradient = std::abs((aParticle.GetMagneticField())[i]);
        }
    }
    aValue = aParticle.GetMagneticField().Magnitude() / MaxGradient * fFraction;
    // std::cout << aValue << "\n";
    return;
}
void KSTrajControlBChange::Check(const KSTrajAdiabaticSpinParticle&, const KSTrajAdiabaticSpinParticle&,
                                 const KSTrajAdiabaticSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlBChange::Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue)
{
    double MaxGradient = 0.;
    for (int i = 0; i < 9; i++) {
        if (std::abs((aParticle.GetMagneticField())[i]) > MaxGradient) {
            MaxGradient = std::abs((aParticle.GetMagneticField())[i]);
        }
    }
    aValue = aParticle.GetMagneticField().Magnitude() / MaxGradient * fFraction;
    return;
}
void KSTrajControlBChange::Check(const KSTrajAdiabaticParticle&, const KSTrajAdiabaticParticle&,
                                 const KSTrajAdiabaticError&, bool& aFlag)
{
    aFlag = true;
    return;
}

}  // namespace Kassiopeia

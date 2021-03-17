#include "KSTrajControlLength.h"

namespace Kassiopeia
{

KSTrajControlLength::KSTrajControlLength() : fLength(0.0) {}
KSTrajControlLength::KSTrajControlLength(const KSTrajControlLength& aCopy) : KSComponent(aCopy), fLength(aCopy.fLength)
{}
KSTrajControlLength* KSTrajControlLength::Clone() const
{
    return new KSTrajControlLength(*this);
}
KSTrajControlLength::~KSTrajControlLength() = default;

void KSTrajControlLength::Calculate(const KSTrajExactParticle& aParticle, double& aValue)
{
    double tLongVelocity = aParticle.GetLongVelocity();
    double tTransVelocity = aParticle.GetTransVelocity();
    double tSpeed = sqrt(tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity);
    aValue = fLength / tSpeed;
    return;
}
void KSTrajControlLength::Check(const KSTrajExactParticle&, const KSTrajExactParticle&, const KSTrajExactError&,
                                bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlLength::Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue)
{
    double tLongVelocity = aParticle.GetLongVelocity();
    double tTransVelocity = aParticle.GetTransVelocity();
    double tSpeed = sqrt(tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity);
    aValue = fLength / tSpeed;
    return;
}
void KSTrajControlLength::Check(const KSTrajExactSpinParticle&, const KSTrajExactSpinParticle&,
                                const KSTrajExactSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlLength::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    double tLongVelocity = aParticle.GetLongVelocity();
    double tTransVelocity = aParticle.GetTransVelocity();
    double tSpeed = sqrt(tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity);
    aValue = fLength / tSpeed;
    return;
}
void KSTrajControlLength::Check(const KSTrajAdiabaticSpinParticle&, const KSTrajAdiabaticSpinParticle&,
                                const KSTrajAdiabaticSpinError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlLength::Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue)
{
    double tLongVelocity = aParticle.GetLongVelocity();
    double tTransVelocity = aParticle.GetTransVelocity();
    double tSpeed = sqrt(tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity);
    aValue = fLength / tSpeed;
    return;
}
void KSTrajControlLength::Check(const KSTrajAdiabaticParticle&, const KSTrajAdiabaticParticle&,
                                const KSTrajAdiabaticError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlLength::Calculate(const KSTrajExactTrappedParticle& aParticle, double& aValue)
{
    double tLongVelocity = aParticle.GetLongVelocity();
    double tTransVelocity = aParticle.GetTransVelocity();
    double tSpeed = sqrt(tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity);
    aValue = fLength / tSpeed;
    return;
}
void KSTrajControlLength::Check(const KSTrajExactTrappedParticle&, const KSTrajExactTrappedParticle&,
                                const KSTrajExactTrappedError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlLength::Calculate(const KSTrajMagneticParticle& /*aParticle*/, double& aValue)
{
    double tSpeed = 1.;  // magnetic particles have no defined velocity
    aValue = fLength / tSpeed;
    return;
}
void KSTrajControlLength::Check(const KSTrajMagneticParticle&, const KSTrajMagneticParticle&,
                                const KSTrajMagneticError&, bool& aFlag)
{
    aFlag = true;
    return;
}

void KSTrajControlLength::Calculate(const KSTrajElectricParticle& aParticle, double& aValue)
{
    double tLongVelocity = aParticle.GetLongVelocity();
    double tTransVelocity = aParticle.GetTransVelocity();
    double tSpeed = sqrt(tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity);
    aValue = fLength / tSpeed;
    return;
}
void KSTrajControlLength::Check(const KSTrajElectricParticle&, const KSTrajElectricParticle&,
                                const KSTrajElectricError&, bool& aFlag)
{
    aFlag = true;
    return;
}

}  // namespace Kassiopeia

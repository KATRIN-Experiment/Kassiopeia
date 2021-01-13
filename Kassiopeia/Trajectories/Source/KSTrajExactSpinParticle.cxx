#include "KSTrajExactSpinParticle.h"

#include "KConst.h"
#include "KSTrajectoriesMessage.h"

#include <cmath>
#include <cstdlib>

namespace Kassiopeia
{

//0 is time
//1 is length
//2 is x component of position
//3 is y component of position
//4 is z component of position
//5 is x component of momentum
//6 is y component of momentum
//7 is z component of momentum

KSMagneticField* KSTrajExactSpinParticle::fMagneticFieldCalculator = nullptr;
KSElectricField* KSTrajExactSpinParticle::fElectricFieldCalculator = nullptr;
double KSTrajExactSpinParticle::fMass = 0.;
double KSTrajExactSpinParticle::fCharge = 0.;
double KSTrajExactSpinParticle::fSpinMagnitude = 0.;
double KSTrajExactSpinParticle::fGyromagneticRatio = 0.;

KSTrajExactSpinParticle::KSTrajExactSpinParticle() :
    fTime(0.),
    fLength(0.),
    fPosition(0., 0., 0.),
    fMomentum(0., 0., 0.),
    fVelocity(0., 0., 0.),
    fSpin0(0.),
    fSpin(0., 0., 0.),
    fLorentzFactor(0.),
    fKineticEnergy(0.),

    fMagneticField(0., 0., 0.),
    fElectricField(0., 0., 0.),
    fMagneticGradient(0., 0., 0., 0., 0., 0., 0., 0., 0.),
    fElectricPotential(0.),

    fGuidingCenter(0., 0., 0.),
    fLongMomentum(0.),
    fTransMomentum(0.),
    fLongVelocity(0.),
    fTransVelocity(0.),
    fCyclotronFrequency(0.),
    fSpinPrecessionFrequency(0.),
    fOrbitalMagneticMoment(0.),

    fGetMagneticFieldPtr(&KSTrajExactSpinParticle::RecalculateMagneticField),
    fGetElectricFieldPtr(&KSTrajExactSpinParticle::RecalculateElectricField),
    fGetMagneticGradientPtr(&KSTrajExactSpinParticle::RecalculateMagneticGradient),
    fGetElectricPotentialPtr(&KSTrajExactSpinParticle::RecalculateElectricPotential)
{}
KSTrajExactSpinParticle::KSTrajExactSpinParticle(const KSTrajExactSpinParticle&) = default;
KSTrajExactSpinParticle::~KSTrajExactSpinParticle() = default;

//**********
//assignment
//**********

void KSTrajExactSpinParticle::PullFrom(const KSParticle& aParticle)
{
    //trajmsg_debug( "exact particle pulling from particle:" << ret )

    if (fMagneticFieldCalculator != aParticle.GetMagneticFieldCalculator()) {
        //trajmsg_debug( "  magnetic calculator differs" << ret )
        fMagneticFieldCalculator = aParticle.GetMagneticFieldCalculator();

        fGetMagneticFieldPtr = &KSTrajExactSpinParticle::RecalculateMagneticField;
        fGetMagneticGradientPtr = &KSTrajExactSpinParticle::RecalculateMagneticGradient;
    }

    if (fElectricFieldCalculator != aParticle.GetElectricFieldCalculator()) {
        //trajmsg_debug( "  electric calculator differs" << ret )
        fElectricFieldCalculator = aParticle.GetElectricFieldCalculator();

        fGetElectricFieldPtr = &KSTrajExactSpinParticle::RecalculateElectricField;
        fGetElectricPotentialPtr = &KSTrajExactSpinParticle::RecalculateElectricPotential;
    }

    if (GetMass() != aParticle.GetMass()) {
        //trajmsg_debug( "  mass differs" << ret )
        fMass = aParticle.GetMass();
    }

    if (GetCharge() != aParticle.GetCharge()) {
        //trajmsg_debug( "  charge differs" << ret )
        fCharge = aParticle.GetCharge();
    }

    if (GetSpinMagnitude() != aParticle.GetSpinMagnitude()) {
        //trajmsg_debug( "  charge differs" << ret )
        fSpinMagnitude = aParticle.GetSpinMagnitude();
    }
    if (GetGyromagneticRatio() != aParticle.GetGyromagneticRatio()) {
        //trajmsg_debug( "  charge differs" << ret )
        fGyromagneticRatio = aParticle.GetGyromagneticRatio();
    }

    if (GetTime() != aParticle.GetTime() || GetPosition() != aParticle.GetPosition()) {
        //trajmsg_debug( "  time or position differs" << ret )

        fTime = aParticle.GetTime();
        fLength = aParticle.GetLength();
        fPosition = aParticle.GetPosition();

        fData[0] = fTime;
        fData[1] = fLength;
        fData[2] = fPosition.X();
        fData[3] = fPosition.Y();
        fData[4] = fPosition.Z();

        fGetMagneticFieldPtr = &KSTrajExactSpinParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajExactSpinParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajExactSpinParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajExactSpinParticle::RecalculateElectricPotential;
    }

    if (GetMomentum() != aParticle.GetMomentum()) {
        //trajmsg_debug( "  momentum differs" << ret )
        fMomentum = aParticle.GetMomentum();

        fData[5] = fMomentum.X();
        fData[6] = fMomentum.Y();
        fData[7] = fMomentum.Z();
    }

    if (GetSpin0() != aParticle.GetSpin0() || GetSpin() != aParticle.GetSpin()) {
        //trajmsg_debug( "  spin differs" << ret )
        fSpin0 = aParticle.GetSpin0();
        fSpin = aParticle.GetSpin();

        fData[8] = fSpin0 / std::sqrt(-fSpin0 * fSpin0 + fSpin.MagnitudeSquared());
        fData[9] = fSpin.X() / std::sqrt(-fSpin0 * fSpin0 + fSpin.MagnitudeSquared());
        fData[10] = fSpin.Y() / std::sqrt(-fSpin0 * fSpin0 + fSpin.MagnitudeSquared());
        fData[11] = fSpin.Z() / std::sqrt(-fSpin0 * fSpin0 + fSpin.MagnitudeSquared());
    }

    //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
    //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
    //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
    //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

    return;
}
void KSTrajExactSpinParticle::PushTo(KSParticle& aParticle) const
{
    //trajmsg_debug( "exact particle pushing to particle:" << eom )


    aParticle.SetLength(GetLength());
    aParticle.SetPosition(GetPosition());
    aParticle.SetMomentum(GetMomentum());
    aParticle.SetSpin0(GetSpin0());
    aParticle.SetSpin(GetSpin());
    aParticle.SetTime(GetTime());

    if (fGetMagneticFieldPtr == &KSTrajExactSpinParticle::DoNothing) {
        aParticle.SetMagneticField(GetMagneticField());
    }
    if (fGetElectricFieldPtr == &KSTrajExactSpinParticle::DoNothing) {
        aParticle.SetElectricField(GetElectricField());
    }
    if (fGetMagneticGradientPtr == &KSTrajExactSpinParticle::DoNothing) {
        aParticle.SetMagneticGradient(GetMagneticGradient());
    }
    if (fGetElectricPotentialPtr == &KSTrajExactSpinParticle::DoNothing) {
        aParticle.SetElectricPotential(GetElectricPotential());
    }

    //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
    //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
    //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
    //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

    return;
}

//***********
//calculators
//***********

void KSTrajExactSpinParticle::SetMagneticFieldCalculator(KSMagneticField* anMagneticField)
{
    fMagneticFieldCalculator = anMagneticField;
    return;
}
KSMagneticField* KSTrajExactSpinParticle::GetMagneticFieldCalculator()
{
    return fMagneticFieldCalculator;
}

void KSTrajExactSpinParticle::SetElectricFieldCalculator(KSElectricField* anElectricField)
{
    fElectricFieldCalculator = anElectricField;
    return;
}
KSElectricField* KSTrajExactSpinParticle::GetElectricFieldCalculator()
{
    return fElectricFieldCalculator;
}

//****************
//static variables
//****************

void KSTrajExactSpinParticle::SetMass(const double& aMass)
{
    fMass = aMass;
    return;
}
const double& KSTrajExactSpinParticle::GetMass()
{
    return fMass;
}

void KSTrajExactSpinParticle::SetCharge(const double& aCharge)
{
    fCharge = aCharge;
    return;
}
const double& KSTrajExactSpinParticle::GetCharge()
{
    return fCharge;
}

void KSTrajExactSpinParticle::SetSpinMagnitude(const double& aSpinMagnitude)
{
    fSpinMagnitude = aSpinMagnitude;
    return;
}
const double& KSTrajExactSpinParticle::GetSpinMagnitude()
{
    return fSpinMagnitude;
}

void KSTrajExactSpinParticle::SetGyromagneticRatio(const double& aGyromagneticRatio)
{
    fGyromagneticRatio = aGyromagneticRatio;
    return;
}
const double& KSTrajExactSpinParticle::GetGyromagneticRatio()
{
    return fGyromagneticRatio;
}

//*****************
//dynamic variables
//*****************

const double& KSTrajExactSpinParticle::GetTime() const
{
    fTime = fData[0];
    return fTime;
}
const double& KSTrajExactSpinParticle::GetLength() const
{
    fLength = fData[1];
    return fLength;
}
const KGeoBag::KThreeVector& KSTrajExactSpinParticle::GetPosition() const
{
    fPosition.SetComponents(fData[2], fData[3], fData[4]);
    return fPosition;
}
const KGeoBag::KThreeVector& KSTrajExactSpinParticle::GetMomentum() const
{
    fMomentum.SetComponents(fData[5], fData[6], fData[7]);
    return fMomentum;
}
const KGeoBag::KThreeVector& KSTrajExactSpinParticle::GetVelocity() const
{
    fVelocity = (1. / (GetMass() * GetLorentzFactor())) * GetMomentum();
    return fVelocity;
}
const double& KSTrajExactSpinParticle::GetSpin0() const
{
    fSpin0 = fData[8];
    return fSpin0;
}
const KGeoBag::KThreeVector& KSTrajExactSpinParticle::GetSpin() const
{
    fSpin.SetComponents(fData[9], fData[10], fData[11]);
    return fSpin;
}
const double& KSTrajExactSpinParticle::GetLorentzFactor() const
{
    fLorentzFactor = sqrt(1. + GetMomentum().MagnitudeSquared() /
                                   (GetMass() * GetMass() * katrin::KConst::C() * katrin::KConst::C()));
    return fLorentzFactor;
}
const double& KSTrajExactSpinParticle::GetKineticEnergy() const
{
    fKineticEnergy = GetMomentum().MagnitudeSquared() / ((1. + GetLorentzFactor()) * fMass);
    return fKineticEnergy;
}

const KGeoBag::KThreeVector& KSTrajExactSpinParticle::GetMagneticField() const
{
    (this->*fGetMagneticFieldPtr)();
    return fMagneticField;
}
const KGeoBag::KThreeVector& KSTrajExactSpinParticle::GetElectricField() const
{
    (this->*fGetElectricFieldPtr)();
    return fElectricField;
}
const KGeoBag::KThreeMatrix& KSTrajExactSpinParticle::GetMagneticGradient() const
{
    (this->*fGetMagneticGradientPtr)();
    return fMagneticGradient;
}
const double& KSTrajExactSpinParticle::GetElectricPotential() const
{
    (this->*fGetElectricPotentialPtr)();
    return fElectricPotential;
}

const KGeoBag::KThreeVector& KSTrajExactSpinParticle::GetGuidingCenter() const
{
    fGuidingCenter = GetPosition() + (1. / (GetCharge() * GetMagneticField().MagnitudeSquared())) *
                                         (GetMomentum().Cross(GetMagneticField()));
    return fGuidingCenter;
}
const double& KSTrajExactSpinParticle::GetLongMomentum() const
{
    fLongMomentum = GetMomentum().Dot(GetMagneticField().Unit());
    return fLongMomentum;
}
const double& KSTrajExactSpinParticle::GetTransMomentum() const
{
    fTransMomentum =
        (GetMomentum() - GetMomentum().Dot(GetMagneticField().Unit()) * GetMagneticField().Unit()).Magnitude();
    return fTransMomentum;
}
const double& KSTrajExactSpinParticle::GetLongVelocity() const
{
    fLongVelocity = GetLongMomentum() / (GetMass() * GetLorentzFactor());
    return fLongVelocity;
}
const double& KSTrajExactSpinParticle::GetTransVelocity() const
{
    fTransVelocity = GetTransMomentum() / (GetMass() * GetLorentzFactor());
    return fTransVelocity;
}
const double& KSTrajExactSpinParticle::GetCyclotronFrequency() const
{
    fCyclotronFrequency =
        (fabs(fCharge) * GetMagneticField().Magnitude()) / (2. * katrin::KConst::Pi() * GetLorentzFactor() * GetMass());
    return fCyclotronFrequency;
}
const double& KSTrajExactSpinParticle::GetSpinPrecessionFrequency() const
{
    fSpinPrecessionFrequency = std::abs(GetGyromagneticRatio() * GetMagneticField().Magnitude());
    return fSpinPrecessionFrequency;
}
const double& KSTrajExactSpinParticle::GetOrbitalMagneticMoment() const
{
    fOrbitalMagneticMoment =
        (GetTransMomentum() * GetTransMomentum()) / (2. * GetMagneticField().Magnitude() * GetMass());
    return fOrbitalMagneticMoment;
}

//*****
//cache
//*****

void KSTrajExactSpinParticle::DoNothing() const
{
    return;
}
void KSTrajExactSpinParticle::RecalculateMagneticField() const
{
    fMagneticFieldCalculator->CalculateField(GetPosition(), GetTime(), fMagneticField);
    fGetMagneticFieldPtr = &KSTrajExactSpinParticle::DoNothing;
    return;
}
void KSTrajExactSpinParticle::RecalculateElectricField() const
{
    fElectricFieldCalculator->CalculateField(GetPosition(), GetTime(), fElectricField);
    fGetElectricFieldPtr = &KSTrajExactSpinParticle::DoNothing;
    return;
}
void KSTrajExactSpinParticle::RecalculateMagneticGradient() const
{
    fMagneticFieldCalculator->CalculateGradient(GetPosition(), GetTime(), fMagneticGradient);
    fGetMagneticGradientPtr = &KSTrajExactSpinParticle::DoNothing;
    return;
}
void KSTrajExactSpinParticle::RecalculateElectricPotential() const
{
    fElectricFieldCalculator->CalculatePotential(GetPosition(), GetTime(), fElectricPotential);
    fGetElectricPotentialPtr = &KSTrajExactSpinParticle::DoNothing;
    return;
}

}  // namespace Kassiopeia

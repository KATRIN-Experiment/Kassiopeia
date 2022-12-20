#include "KSTrajElectricParticle.h"

#include "KConst.h"
#include "KSTrajectoriesMessage.h"

#include <cmath>

using katrin::KThreeMatrix;
using katrin::KThreeVector;

namespace Kassiopeia
{

//0 is time
//1 is length
//2 is x component of position
//3 is y component of position
//4 is z component of position

KSMagneticField* KSTrajElectricParticle::fMagneticFieldCalculator = nullptr;
KSElectricField* KSTrajElectricParticle::fElectricFieldCalculator = nullptr;
double KSTrajElectricParticle::fMass = 0.;
double KSTrajElectricParticle::fCharge = 0.;

KSTrajElectricParticle::KSTrajElectricParticle() :
    fTime(0.),
    fLength(0.),
    fPosition(0., 0., 0.),
    fMomentum(0., 0., 0.),
    fVelocity(0., 0., 0.),
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
    fOrbitalMagneticMoment(0.),

    fGetMagneticFieldPtr(&KSTrajElectricParticle::RecalculateMagneticField),
    fGetElectricFieldPtr(&KSTrajElectricParticle::RecalculateElectricField),
    fGetMagneticGradientPtr(&KSTrajElectricParticle::RecalculateMagneticGradient),
    fGetElectricPotentialPtr(&KSTrajElectricParticle::RecalculateElectricPotential)
{}
KSTrajElectricParticle::KSTrajElectricParticle(const KSTrajElectricParticle&) = default;
KSTrajElectricParticle::~KSTrajElectricParticle() = default;

//**********
//assignment
//**********

void KSTrajElectricParticle::PullFrom(const KSParticle& aParticle)
{
    //trajmsg_debug( "magnetic particle pulling from particle:" << ret )

    if (fMagneticFieldCalculator != aParticle.GetMagneticFieldCalculator()) {
        //trajmsg_debug( "  magnetic calculator differs" << ret )
        fMagneticFieldCalculator = aParticle.GetMagneticFieldCalculator();

        fGetMagneticFieldPtr = &KSTrajElectricParticle::RecalculateMagneticField;
        fGetMagneticGradientPtr = &KSTrajElectricParticle::RecalculateMagneticGradient;
    }

    if (fElectricFieldCalculator != aParticle.GetElectricFieldCalculator()) {
        //trajmsg_debug( "  electric calculator differs" << ret )
        fElectricFieldCalculator = aParticle.GetElectricFieldCalculator();

        fGetElectricFieldPtr = &KSTrajElectricParticle::RecalculateElectricField;
        fGetElectricPotentialPtr = &KSTrajElectricParticle::RecalculateElectricPotential;
    }

    if (GetMass() != aParticle.GetMass()) {
        //trajmsg_debug( "  mass differs" << ret )
        fMass = aParticle.GetMass();
    }

    if (GetCharge() != aParticle.GetCharge()) {
        //trajmsg_debug( "  charge differs" << ret )
        fCharge = aParticle.GetCharge();
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

        fGetMagneticFieldPtr = &KSTrajElectricParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajElectricParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajElectricParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajElectricParticle::RecalculateElectricPotential;
    }

    //trajmsg_debug( "  time: <" << GetTime() << ">" << eom )
    //trajmsg_debug( "  length: <" << GetLength() << ">" << eom )
    //trajmsg_debug( "  position: <" << GetPosition().X() << ", " << GetPosition().Y() << ", " << GetPosition().Z() << ">" << eom )
    //trajmsg_debug( "  momentum: <"  << GetMomentum().X() << ", " << GetMomentum().Y() << ", " << GetMomentum().Z() << ">" << eom )

    return;
}
void KSTrajElectricParticle::PushTo(KSParticle& aParticle)
{
    //trajmsg_debug( "magnetic particle pushing to particle:" << eom )

    aParticle.SetTime(GetTime());
    aParticle.SetLength(GetLength());
    aParticle.SetPosition(GetPosition());
    aParticle.SetMomentum(GetMomentum());

    if (fGetMagneticFieldPtr == &KSTrajElectricParticle::DoNothing) {
        aParticle.SetMagneticField(GetMagneticField());
    }
    if (fGetElectricFieldPtr == &KSTrajElectricParticle::DoNothing) {
        aParticle.SetElectricField(GetElectricField());
    }
    if (fGetMagneticGradientPtr == &KSTrajElectricParticle::DoNothing) {
        aParticle.SetMagneticGradient(GetMagneticGradient());
    }
    if (fGetElectricPotentialPtr == &KSTrajElectricParticle::DoNothing) {
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

void KSTrajElectricParticle::SetMagneticFieldCalculator(KSMagneticField* anMagneticField)
{
    fMagneticFieldCalculator = anMagneticField;
    return;
}
KSMagneticField* KSTrajElectricParticle::GetMagneticFieldCalculator()
{
    return fMagneticFieldCalculator;
}

void KSTrajElectricParticle::SetElectricFieldCalculator(KSElectricField* anElectricField)
{
    fElectricFieldCalculator = anElectricField;
    return;
}
KSElectricField* KSTrajElectricParticle::GetElectricFieldCalculator()
{
    return fElectricFieldCalculator;
}

//****************
//static variables
//****************

void KSTrajElectricParticle::SetMass(const double& aMass)
{
    fMass = aMass;
    return;
}
const double& KSTrajElectricParticle::GetMass()
{
    return fMass;
}

void KSTrajElectricParticle::SetCharge(const double& aCharge)
{
    fCharge = aCharge;
    return;
}
const double& KSTrajElectricParticle::GetCharge()
{
    return fCharge;
}

//*****************
//dynamic variables
//*****************

const double& KSTrajElectricParticle::GetTime() const
{
    fTime = fData[0];
    return fTime;
}
const double& KSTrajElectricParticle::GetLength() const
{
    fLength = fData[1];
    return fLength;
}
const KThreeVector& KSTrajElectricParticle::GetPosition() const
{
    fPosition.SetComponents(fData[2], fData[3], fData[4]);
    return fPosition;
}
const KThreeVector& KSTrajElectricParticle::GetMomentum() const
{
    fMomentum = GetMass() * GetVelocity() * GetLorentzFactor();
    return fMomentum;
}
const KThreeVector& KSTrajElectricParticle::GetVelocity() const
{
    fVelocity = GetElectricField().Unit();
    return fVelocity;
}
const double& KSTrajElectricParticle::GetLorentzFactor() const
{
    fLorentzFactor = 1. / sqrt(1. - GetVelocity().MagnitudeSquared() / (katrin::KConst::C() * katrin::KConst::C()));
    return fLorentzFactor;
}
const double& KSTrajElectricParticle::GetKineticEnergy() const
{
    fKineticEnergy = GetMomentum().MagnitudeSquared() / ((1. + GetLorentzFactor()) * fMass);
    return fKineticEnergy;
}

const KThreeVector& KSTrajElectricParticle::GetMagneticField() const
{
    (this->*fGetMagneticFieldPtr)();
    return fMagneticField;
}
const KThreeVector& KSTrajElectricParticle::GetElectricField() const
{
    (this->*fGetElectricFieldPtr)();
    return fElectricField;
}
const KThreeMatrix& KSTrajElectricParticle::GetMagneticGradient() const
{
    (this->*fGetMagneticGradientPtr)();
    return fMagneticGradient;
}
const double& KSTrajElectricParticle::GetElectricPotential() const
{
    (this->*fGetElectricPotentialPtr)();
    return fElectricPotential;
}

const KThreeVector& KSTrajElectricParticle::GetGuidingCenter() const
{
    fGuidingCenter = GetPosition();
    return fGuidingCenter;
}
const double& KSTrajElectricParticle::GetLongMomentum() const
{
    fLongMomentum = GetMomentum().Magnitude();
    return fLongMomentum;
}
const double& KSTrajElectricParticle::GetTransMomentum() const
{
    fTransMomentum = 0.;
    return fTransMomentum;
}
const double& KSTrajElectricParticle::GetLongVelocity() const
{
    fLongVelocity = GetVelocity().Magnitude();
    return fLongVelocity;
}
const double& KSTrajElectricParticle::GetTransVelocity() const
{
    fTransVelocity = 0.;
    return fTransVelocity;
}
const double& KSTrajElectricParticle::GetCyclotronFrequency() const
{
    fCyclotronFrequency =
        (fabs(fCharge) * GetMagneticField().Magnitude()) / (2. * katrin::KConst::Pi() * GetLorentzFactor() * GetMass());
    return fCyclotronFrequency;
}
const double& KSTrajElectricParticle::GetOrbitalMagneticMoment() const
{
    fOrbitalMagneticMoment = 0.;
    return fOrbitalMagneticMoment;
}

//*****
//cache
//*****

void KSTrajElectricParticle::DoNothing() const
{
    return;
}
void KSTrajElectricParticle::RecalculateMagneticField() const
{
    fMagneticFieldCalculator->CalculateField(GetPosition(), GetTime(), fMagneticField);
    fGetMagneticFieldPtr = &KSTrajElectricParticle::DoNothing;
    return;
}
void KSTrajElectricParticle::RecalculateElectricField() const
{
    fElectricFieldCalculator->CalculateField(GetPosition(), GetTime(), fElectricField);
    fGetElectricFieldPtr = &KSTrajElectricParticle::DoNothing;
    return;
}
void KSTrajElectricParticle::RecalculateMagneticGradient() const
{
    fMagneticFieldCalculator->CalculateGradient(GetPosition(), GetTime(), fMagneticGradient);
    fGetMagneticGradientPtr = &KSTrajElectricParticle::DoNothing;
    return;
}
void KSTrajElectricParticle::RecalculateElectricPotential() const
{
    fElectricFieldCalculator->CalculatePotential(GetPosition(), GetTime(), fElectricPotential);
    fGetElectricPotentialPtr = &KSTrajElectricParticle::DoNothing;
    return;
}

}  // namespace Kassiopeia

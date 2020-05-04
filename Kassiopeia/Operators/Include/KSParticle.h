#ifndef Kassiopeia_KSParticle_h_
#define Kassiopeia_KSParticle_h_

#include "KSElectricField.h"
#include "KSMagneticField.h"
#include "KSSide.h"
#include "KSSpace.h"
#include "KSSurface.h"
#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

#include <deque>

namespace Kassiopeia
{

class KSParticleFactory;

class KSParticle
{
  public:
    friend class KSParticleFactory;

    //**********
    //assignment
    //**********

  public:
    KSParticle();
    KSParticle(const KSParticle& aParticleToClone);
    void operator=(const KSParticle& aParticle);
    ~KSParticle();

    void Print() const;
    void DoNothing() const;
    bool IsValid() const;

    //******
    //labels
    //******

  public:
    static const std::string sSeparator;

    const std::string& GetLabel() const;
    void SetLabel(const std::string& aLabel);
    void AddLabel(const std::string& aLabel);
    void ReleaseLabel(std::string& aLabel);

    void SetIndexNumber(const long& anId);
    const long& GetIndexNumber() const;

    void SetParentRunId(const int& anId);
    const int& GetParentRunId() const;

    void SetParentEventId(const int& anId);
    const int& GetParentEventId() const;

    void SetParentTrackId(const int& anId);
    const int& GetParentTrackId() const;

    void SetParentStepId(const int& anId);
    const int& GetParentStepId() const;

  private:
    std::string fLabel;
    long fIndexNumber;
    int fParentRunId;
    int fParentEventId;
    int fParentTrackId;
    int fParentStepId;

    //*****
    //state
    //*****

  public:
    void SetActive(const bool&);
    const bool& IsActive() const;

    void SetCurrentSpace(KSSpace* aSpace);
    KSSpace* GetCurrentSpace() const;
    const std::string& GetCurrentSpaceName() const;

    void SetCurrentSurface(KSSurface* aSurface);
    KSSurface* GetCurrentSurface() const;
    const std::string& GetCurrentSurfaceName() const;

    void SetCurrentSide(KSSide* aSide);
    KSSide* GetCurrentSide() const;
    const std::string& GetCurrentSideName() const;

    void SetLastStepSurface(KSSurface* aSurface);
    KSSurface* GetLastStepSurface() const;

  private:
    bool fActive;
    KSSpace* fCurrentSpace;
    KSSurface* fCurrentSurface;
    KSSurface* fLastStepSurface;
    KSSide* fCurrentSide;
    std::string fCurrentSpaceName;
    std::string fCurrentSurfaceName;
    std::string fCurrentSideName;

    //***********
    //calculators
    //***********

  public:
    void SetMagneticFieldCalculator(KSMagneticField* aCalculator);
    KSMagneticField* GetMagneticFieldCalculator() const;

    void SetElectricFieldCalculator(KSElectricField* aCalculator);
    KSElectricField* GetElectricFieldCalculator() const;

    void ResetFieldCaching();

  protected:
    KSMagneticField* fMagneticFieldCalculator;
    KSElectricField* fElectricFieldCalculator;

    //*****************
    //static properties
    //*****************

  public:
    const long long& GetPID() const;
    const std::string& GetStringID() const;
    const double& GetMass() const;
    const double& GetCharge() const;
    const double& GetSpinMagnitude() const;
    const double& GetGyromagneticRatio() const;

  protected:
    long long fPID;
    std::string fStringID;
    double fMass;               // in kg
    double fCharge;             // in Coulomb
    double fSpinMagnitude;      // in hbar
    double fGyromagneticRatio;  // in rad/sT

    //******************
    //dynamic properties
    //******************

  public:
    // quantum numbers

    const int& GetMainQuantumNumber() const;

    void SetMainQuantumNumber(const int& t);

    const int& GetSecondQuantumNumber() const;

    void SetSecondQuantumNumber(const int& t);


    //time (units are seconds)

    const double& GetTime() const;

    void SetTime(const double& t);

    //length (units are meters)

    const double& GetLength() const;

    void SetLength(const double& l);

    //position (units are meters)

    const KThreeVector& GetPosition() const;
    double GetX() const;
    double GetY() const;
    double GetZ() const;

    void SetPosition(const KThreeVector& position);
    void SetPosition(const double& x, const double& y, const double& z);
    void SetX(const double& x);
    void SetY(const double& y);
    void SetZ(const double& z);

    void RecalculatePosition() const;

    //momentum (units are kg*m/s)

    const KThreeVector& GetMomentum() const;
    double GetPX() const;
    double GetPY() const;
    double GetPZ() const;

    void SetMomentum(const KThreeVector& momentum);
    void SetMomentum(const double& px, const double& py, const double& pz);
    void SetPX(const double& px);
    void SetPY(const double& py);
    void SetPZ(const double& pz);

    void RecalculateMomentum() const;

    //velocity (units are m/s)

    const KThreeVector& GetVelocity() const;

    void SetVelocity(const KThreeVector& velocity);

    void RecalculateVelocity() const;

    //spin0

    const double& GetSpin0() const;

    void SetSpin0(const double& spin0);

    void RecalculateSpin0() const;

    //spin

    const KThreeVector& GetSpin() const;
    double GetSpinX() const;
    double GetSpinY() const;
    double GetSpinZ() const;

    void SetSpin(const KThreeVector& spin);
    void SetSpin(const double& spinx, const double& spiny, const double& spinz);
    void SetSpinX(const double& spinx);
    void SetSpinY(const double& spiny);
    void SetSpinZ(const double& spinz);

    void SetInitialSpin(const KThreeVector& spin);

    void NormalizeSpin() const;

    void RecalculateSpin() const;

    void RecalculateSpinGlobal() const;

    // lorentz factor (unitless)

    const double& GetLorentzFactor() const;

    void SetLorentzFactor(const double& lorentzfactor);

    void RecalculateLorentzFactor() const;

    //speed (units are m/s)

    const double& GetSpeed() const;

    void SetSpeed(const double& speed);

    void RecalculateSpeed() const;

    //kinetic energy (units are J)

    const double& GetKineticEnergy() const;

    void SetKineticEnergy(const double& energy);

    void RecalculateKineticEnergy() const;

    //kinetic energy (units are eV)

    const double& GetKineticEnergy_eV() const;

    void SetKineticEnergy_eV(const double& energy);

    //polar angle of momentum std::vector (units are degrees)

    const double& GetPolarAngleToZ() const;

    void SetPolarAngleToZ(const double& NewPolarAngleToZ);

    void RecalculatePolarAngleToZ() const;

    //azimuthal angle of momentum std::vector (units are degrees)

    const double& GetAzimuthalAngleToX() const;

    void SetAzimuthalAngleToX(const double& NewAzimuthalAngleToX);

    void RecalculateAzimuthalAngleToX() const;

  protected:
    mutable int fMainQuantumNumber;
    mutable int fSecondQuantumNumber;
    mutable double fTime;
    mutable double fLength;
    mutable KThreeVector fPosition;
    mutable KThreeVector fMomentum;
    mutable KThreeVector fVelocity;
    mutable double fSpin0;
    mutable KThreeVector fSpin;
    mutable double fLorentzFactor;
    mutable double fSpeed;
    mutable double fKineticEnergy;
    mutable double fKineticEnergy_eV;
    mutable double fPolarAngleToZ;
    mutable double fAzimuthalAngleToX;

  protected:
    mutable void (KSParticle::*fGetPositionAction)() const;
    mutable void (KSParticle::*fGetMomentumAction)() const;
    mutable void (KSParticle::*fGetVelocityAction)() const;
    //mutable void (KSParticle::*fGetSpin0Action)() const;
    //mutable void (KSParticle::*fGetSpinAction)() const;
    mutable void (KSParticle::*fGetLorentzFactorAction)() const;
    mutable void (KSParticle::*fGetSpeedAction)() const;
    mutable void (KSParticle::*fGetKineticEnergyAction)() const;
    mutable void (KSParticle::*fGetPolarAngleToZAction)() const;
    mutable void (KSParticle::*fGetAzimuthalAngleToXAction)() const;

    //**************************
    //electromagnetic properties
    //**************************

  public:
    //magnetic field (units are tesla)

    const KThreeVector& GetMagneticField() const;

    void SetMagneticField(const KThreeVector&);

    void RecalculateMagneticField() const;

    //electric field (units are volt/meter)

    const KThreeVector& GetElectricField() const;

    void SetElectricField(const KThreeVector&);

    void RecalculateElectricField() const;

    //gradient of magnetic field (units are tesla/meter)

    const KThreeMatrix& GetMagneticGradient() const;

    void SetMagneticGradient(const KThreeMatrix&);

    void RecalculateMagneticGradient() const;

    //electric potential (units are volt)

    const double& GetElectricPotential() const;

    void SetElectricPotential(const double&);

    void RecalculateElectricPotential() const;

  protected:
    mutable KThreeVector fMagneticField;
    mutable KThreeVector fElectricField;
    mutable KThreeMatrix fMagneticGradient;
    mutable double fElectricPotential;

  protected:
    mutable void (KSParticle::*fGetMagneticFieldAction)() const;
    mutable void (KSParticle::*fGetElectricFieldAction)() const;
    mutable void (KSParticle::*fGetMagneticGradientAction)() const;
    mutable void (KSParticle::*fGetElectricPotentialAction)() const;

    //**********************************
    //electromagnetic dynamic properties
    //**********************************

  public:
    //longitudinal momentum (units are kg*m/s)

    const double& GetLongMomentum() const;

    void SetLongMomentum(const double&);

    void RecalculateLongMomentum() const;

    //transverse momentum (units are kg*m/s)

    const double& GetTransMomentum() const;

    void SetTransMomentum(const double&);

    void RecalculateTransMomentum() const;

    //longitudinal velocity (units are m/s)

    const double& GetLongVelocity() const;

    void SetLongVelocity(const double&);

    void RecalculateLongVelocity() const;

    //transverse velocity (units are m/s)

    const double& GetTransVelocity() const;

    void SetTransVelocity(const double&);

    void RecalculateTransVelocity() const;

    //polar angle to b (units are degrees)

    const double& GetPolarAngleToB() const;

    void SetPolarAngleToB(const double&);

    void RecalculatePolarAngleToB() const;

    //cyclotron frequency (units are 1/s)

    const double& GetCyclotronFrequency() const;

    void SetCyclotronFrequency(const double&);

    void RecalculateCyclotronFrequency() const;

    //orbital magnetic moment (units are A*m^2)

    const double& GetOrbitalMagneticMoment() const;

    void SetOrbitalMagneticMoment(const double&);

    void RecalculateOrbitalMagneticMoment() const;

    //guiding center position

    const KThreeVector& GetGuidingCenterPosition() const;

    void SetGuidingCenterPosition(const KThreeVector& aPosition);

    void RecalculateGuidingCenterPosition() const;

  protected:
    mutable double fLongMomentum;
    mutable double fTransMomentum;
    mutable double fLongVelocity;
    mutable double fTransVelocity;
    mutable double fPolarAngleToB;
    mutable double fCyclotronFrequency;
    mutable double fOrbitalMagneticMoment;
    mutable KThreeVector fGuidingCenterPosition;

  protected:
    mutable void (KSParticle::*fGetLongMomentumAction)() const;
    mutable void (KSParticle::*fGetTransMomentumAction)() const;
    mutable void (KSParticle::*fGetLongVelocityAction)() const;
    mutable void (KSParticle::*fGetTransVelocityAction)() const;
    mutable void (KSParticle::*fGetPolarAngleToBAction)() const;
    mutable void (KSParticle::*fGetCyclotronFrequencyAction)() const;
    mutable void (KSParticle::*fGetOrbitalMagneticMomentAction)() const;
    mutable void (KSParticle::*fGetGuidingCenterPositionAction)() const;

  public:
    //spin component along local B

    const double& GetAlignedSpin() const;

    void SetAlignedSpin(const double&) const;

    //spin angle around local B, using <bz-by, bx-bz, by-bx> as the x-axis and B cross x as the y axis

    const double& GetSpinAngle() const;

    void SetSpinAngle(const double&) const;

    void RecalculateSpinBody() const;

  protected:
    mutable double fAlignedSpin;
    mutable double fSpinAngle;

    //clock time (units are seconds)

  public:
    const double& GetClockTime() const;

  protected:
    mutable double fClockTime;
};

typedef std::deque<KSParticle*> KSParticleQueue;
typedef KSParticleQueue::iterator KSParticleIt;
typedef KSParticleQueue::const_iterator KSParticleCIt;

}  // namespace Kassiopeia

#endif

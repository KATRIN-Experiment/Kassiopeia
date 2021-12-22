#ifndef Kassiopeia_KSTrajExactSpinParticle_h_
#define Kassiopeia_KSTrajExactSpinParticle_h_

#include "KSMathArray.h"
#include "KSParticle.h"

namespace Kassiopeia
{

class KSTrajExactSpinParticle : public KSMathArray<12>
{
  public:
    KSTrajExactSpinParticle();
    KSTrajExactSpinParticle(const KSTrajExactSpinParticle& aParticle);
    ~KSTrajExactSpinParticle() override;

    //**********
    //assignment
    //**********

  public:
    void PullFrom(const KSParticle& aParticle);
    void PushTo(KSParticle& aParticle) const;

    KSTrajExactSpinParticle& operator=(const double& anOperand);

    KSTrajExactSpinParticle& operator=(const KSMathArray<12>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajExactSpinParticle& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajExactSpinParticle& operator=(const KSTrajExactSpinParticle& anOperand);

    //***********
    //calculators
    //***********

  public:
    static void SetMagneticFieldCalculator(KSMagneticField* aMagneticField);
    static KSMagneticField* GetMagneticFieldCalculator();

    static void SetElectricFieldCalculator(KSElectricField* anElectricField);
    static KSElectricField* GetElectricFieldCalculator();

    //****************
    //static variables
    //****************

  public:
    static void SetMass(const double& aMass);
    static const double& GetMass();

    static void SetCharge(const double& aCharge);
    static const double& GetCharge();

    static void SetSpinMagnitude(const double& aSpinMagnitude);
    static const double& GetSpinMagnitude();

    static void SetGyromagneticRatio(const double& aGyromagneticRatio);
    static const double& GetGyromagneticRatio();

    //*****************
    //dynamic variables
    //*****************

  public:
    const double& GetTime() const;
    const double& GetLength() const;
    const katrin::KThreeVector& GetPosition() const;
    const katrin::KThreeVector& GetMomentum() const;
    const katrin::KThreeVector& GetVelocity() const;
    const double& GetSpin0() const;
    const katrin::KThreeVector& GetSpin() const;
    const double& GetLorentzFactor() const;
    const double& GetKineticEnergy() const;

    const katrin::KThreeVector& GetMagneticField() const;
    const katrin::KThreeVector& GetElectricField() const;
    const double& GetElectricPotential() const;
    const katrin::KThreeMatrix& GetMagneticGradient() const;

    const katrin::KThreeVector& GetGuidingCenter() const;
    const double& GetLongMomentum() const;
    const double& GetTransMomentum() const;
    const double& GetLongVelocity() const;
    const double& GetTransVelocity() const;
    const double& GetCyclotronFrequency() const;
    const double& GetSpinPrecessionFrequency() const;
    const double& GetOrbitalMagneticMoment() const;

  private:
    static KSMagneticField* fMagneticFieldCalculator;
    static KSElectricField* fElectricFieldCalculator;

    static double fMass;
    static double fCharge;
    static double fSpinMagnitude;
    static double fGyromagneticRatio;

    mutable double fTime;
    mutable double fLength;
    mutable katrin::KThreeVector fPosition;
    mutable katrin::KThreeVector fMomentum;
    mutable katrin::KThreeVector fVelocity;
    mutable double fSpin0;
    mutable katrin::KThreeVector fSpin;
    mutable double fLorentzFactor;
    mutable double fKineticEnergy;

    mutable katrin::KThreeVector fMagneticField;
    mutable katrin::KThreeVector fElectricField;
    mutable katrin::KThreeMatrix fMagneticGradient;
    mutable double fElectricPotential;

    mutable katrin::KThreeVector fGuidingCenter;
    mutable double fLongMomentum;
    mutable double fTransMomentum;
    mutable double fLongVelocity;
    mutable double fTransVelocity;
    mutable double fCyclotronFrequency;
    mutable double fSpinPrecessionFrequency;
    mutable double fOrbitalMagneticMoment;

    //*****
    //cache
    //*****

  private:
    void DoNothing() const;
    void RecalculateMagneticField() const;
    void RecalculateMagneticGradient() const;
    void RecalculateElectricField() const;
    void RecalculateElectricPotential() const;

    mutable void (KSTrajExactSpinParticle::*fGetMagneticFieldPtr)() const;
    mutable void (KSTrajExactSpinParticle::*fGetElectricFieldPtr)() const;
    mutable void (KSTrajExactSpinParticle::*fGetMagneticGradientPtr)() const;
    mutable void (KSTrajExactSpinParticle::*fGetElectricPotentialPtr)() const;
};

inline KSTrajExactSpinParticle& KSTrajExactSpinParticle::operator=(const double& anOperand)
{
    this->KSMathArray<12>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajExactSpinParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajExactSpinParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajExactSpinParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajExactSpinParticle::RecalculateElectricPotential;
    return *this;
}

inline KSTrajExactSpinParticle& KSTrajExactSpinParticle::operator=(const KSMathArray<12>& anOperand)
{
    this->KSMathArray<12>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajExactSpinParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajExactSpinParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajExactSpinParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajExactSpinParticle::RecalculateElectricPotential;
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajExactSpinParticle&
KSTrajExactSpinParticle::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<12>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajExactSpinParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajExactSpinParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajExactSpinParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajExactSpinParticle::RecalculateElectricPotential;
    return *this;
}

inline KSTrajExactSpinParticle& KSTrajExactSpinParticle::operator=(const KSTrajExactSpinParticle& aParticle)
{
    this->KSMathArray<12>::operator=(aParticle);

    fTime = aParticle.fTime;
    fLength = aParticle.fLength;
    fPosition = aParticle.fPosition;
    fMomentum = aParticle.fMomentum;
    fVelocity = aParticle.fVelocity;
    fSpin0 = aParticle.fSpin0;
    fSpin = aParticle.fSpin;
    fLorentzFactor = aParticle.fLorentzFactor;
    fKineticEnergy = aParticle.fKineticEnergy;

    fMagneticField = aParticle.fMagneticField;
    fElectricField = aParticle.fElectricField;
    fMagneticGradient = aParticle.fMagneticGradient;
    fElectricPotential = aParticle.fElectricPotential;

    fGuidingCenter = aParticle.fGuidingCenter;
    fLongMomentum = aParticle.fLongMomentum;
    fTransMomentum = aParticle.fTransMomentum;
    fLongVelocity = aParticle.fLongVelocity;
    fTransVelocity = aParticle.fTransVelocity;
    fCyclotronFrequency = aParticle.fCyclotronFrequency;
    fSpinPrecessionFrequency = aParticle.fSpinPrecessionFrequency;
    fOrbitalMagneticMoment = aParticle.fOrbitalMagneticMoment;

    fGetMagneticFieldPtr = aParticle.fGetMagneticFieldPtr;
    fGetElectricFieldPtr = aParticle.fGetElectricFieldPtr;
    fGetMagneticGradientPtr = aParticle.fGetMagneticGradientPtr;
    fGetElectricPotentialPtr = aParticle.fGetElectricPotentialPtr;

    return *this;
}

}  // namespace Kassiopeia

#endif

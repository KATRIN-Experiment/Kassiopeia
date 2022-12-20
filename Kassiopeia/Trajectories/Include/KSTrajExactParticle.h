#ifndef Kassiopeia_KSTrajExactParticle_h_
#define Kassiopeia_KSTrajExactParticle_h_

#include "KSMathArray.h"
#include "KSParticle.h"

namespace Kassiopeia
{

class KSTrajExactParticle : public KSMathArray<8>
{
  public:
    KSTrajExactParticle();
    KSTrajExactParticle(const KSTrajExactParticle& aParticle);
    ~KSTrajExactParticle() override;

    //**********
    //assignment
    //**********

  public:
    void PullFrom(const KSParticle& aParticle);
    void PushTo(KSParticle& aParticle) const;

    KSTrajExactParticle& operator=(const double& anOperand);

    KSTrajExactParticle& operator=(const KSMathArray<8>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajExactParticle& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajExactParticle& operator=(const KSTrajExactParticle& anOperand);

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

    //*****************
    //dynamic variables
    //*****************

  public:
    const double& GetTime() const;
    const double& GetLength() const;
    const katrin::KThreeVector& GetPosition() const;
    const katrin::KThreeVector& GetMomentum() const;
    const katrin::KThreeVector& GetVelocity() const;
    const double& GetLorentzFactor() const;
    const double& GetKineticEnergy() const;

    const katrin::KThreeVector& GetMagneticField() const;
    const katrin::KThreeVector& GetElectricField() const;
    const double& GetElectricPotential() const;
    const katrin::KThreeMatrix& GetElectricGradient() const;
    const katrin::KThreeMatrix& GetMagneticGradient() const;

    const katrin::KThreeVector& GetGuidingCenter() const;
    const double& GetLongMomentum() const;
    const double& GetTransMomentum() const;
    const double& GetLongVelocity() const;
    const double& GetTransVelocity() const;
    const double& GetCyclotronFrequency() const;
    const double& GetOrbitalMagneticMoment() const;

  private:
    static KSMagneticField* fMagneticFieldCalculator;
    static KSElectricField* fElectricFieldCalculator;

    static double fMass;
    static double fCharge;

    mutable double fTime;
    mutable double fLength;
    mutable katrin::KThreeVector fPosition;
    mutable katrin::KThreeVector fMomentum;
    mutable katrin::KThreeVector fVelocity;
    mutable double fLorentzFactor;
    mutable double fKineticEnergy;

    mutable katrin::KThreeVector fMagneticField;
    mutable katrin::KThreeVector fElectricField;
    mutable katrin::KThreeMatrix fMagneticGradient;
    mutable katrin::KThreeMatrix fElectricGradient;
    mutable double fElectricPotential;

    mutable katrin::KThreeVector fGuidingCenter;
    mutable double fLongMomentum;
    mutable double fTransMomentum;
    mutable double fLongVelocity;
    mutable double fTransVelocity;
    mutable double fCyclotronFrequency;
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
    void RecalculateElectricGradient() const;

    mutable void (KSTrajExactParticle::*fGetMagneticFieldPtr)() const;
    mutable void (KSTrajExactParticle::*fGetElectricFieldPtr)() const;
    mutable void (KSTrajExactParticle::*fGetMagneticGradientPtr)() const;
    mutable void (KSTrajExactParticle::*fGetElectricPotentialPtr)() const;
    mutable void (KSTrajExactParticle::*fGetElectricGradientPtr)() const;
};

inline KSTrajExactParticle& KSTrajExactParticle::operator=(const double& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajExactParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajExactParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajExactParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajExactParticle::RecalculateElectricPotential;
    fGetElectricGradientPtr = &KSTrajExactParticle::RecalculateElectricGradient;
    return *this;
}

inline KSTrajExactParticle& KSTrajExactParticle::operator=(const KSMathArray<8>& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajExactParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajExactParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajExactParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajExactParticle::RecalculateElectricPotential;
    fGetElectricGradientPtr = &KSTrajExactParticle::RecalculateElectricGradient;
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajExactParticle& KSTrajExactParticle::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajExactParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajExactParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajExactParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajExactParticle::RecalculateElectricPotential;
    fGetElectricGradientPtr = &KSTrajExactParticle::RecalculateElectricGradient;
    return *this;
}

inline KSTrajExactParticle& KSTrajExactParticle::operator=(const KSTrajExactParticle& aParticle)
{
    this->KSMathArray<8>::operator=(aParticle);

    fTime = aParticle.fTime;
    fLength = aParticle.fLength;
    fPosition = aParticle.fPosition;
    fMomentum = aParticle.fMomentum;
    fVelocity = aParticle.fVelocity;
    fLorentzFactor = aParticle.fLorentzFactor;
    fKineticEnergy = aParticle.fKineticEnergy;

    fMagneticField = aParticle.fMagneticField;
    fElectricField = aParticle.fElectricField;
    fMagneticGradient = aParticle.fMagneticGradient;
    fElectricPotential = aParticle.fElectricPotential;
    fElectricGradient = aParticle.fElectricGradient;

    fGuidingCenter = aParticle.fGuidingCenter;
    fLongMomentum = aParticle.fLongMomentum;
    fTransMomentum = aParticle.fTransMomentum;
    fLongVelocity = aParticle.fLongVelocity;
    fTransVelocity = aParticle.fTransVelocity;
    fCyclotronFrequency = aParticle.fCyclotronFrequency;
    fOrbitalMagneticMoment = aParticle.fOrbitalMagneticMoment;

    fGetMagneticFieldPtr = aParticle.fGetMagneticFieldPtr;
    fGetElectricFieldPtr = aParticle.fGetElectricFieldPtr;
    fGetMagneticGradientPtr = aParticle.fGetMagneticGradientPtr;
    fGetElectricPotentialPtr = aParticle.fGetElectricPotentialPtr;
    fGetElectricGradientPtr = aParticle.fGetElectricGradientPtr;

    return *this;
}

}  // namespace Kassiopeia

#endif

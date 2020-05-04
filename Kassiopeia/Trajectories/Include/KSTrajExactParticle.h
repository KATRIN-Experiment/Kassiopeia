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
    ~KSTrajExactParticle();

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
    const KThreeVector& GetPosition() const;
    const KThreeVector& GetMomentum() const;
    const KThreeVector& GetVelocity() const;
    const double& GetLorentzFactor() const;
    const double& GetKineticEnergy() const;

    const KThreeVector& GetMagneticField() const;
    const KThreeVector& GetElectricField() const;
    const double& GetElectricPotential() const;
    const KThreeMatrix& GetElectricGradient() const;
    const KThreeMatrix& GetMagneticGradient() const;

    const KThreeVector& GetGuidingCenter() const;
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
    mutable KThreeVector fPosition;
    mutable KThreeVector fMomentum;
    mutable KThreeVector fVelocity;
    mutable double fLorentzFactor;
    mutable double fKineticEnergy;

    mutable KThreeVector fMagneticField;
    mutable KThreeVector fElectricField;
    mutable KThreeMatrix fMagneticGradient;
    mutable KThreeMatrix fElectricGradient;
    mutable double fElectricPotential;

    mutable KThreeVector fGuidingCenter;
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

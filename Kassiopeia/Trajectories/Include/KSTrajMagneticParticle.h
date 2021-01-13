#ifndef Kassiopeia_KSTrajMagneticParticle_h_
#define Kassiopeia_KSTrajMagneticParticle_h_

#include "KSMathArray.h"
#include "KSParticle.h"

namespace Kassiopeia
{

class KSTrajMagneticParticle : public KSMathArray<5>
{
  public:
    KSTrajMagneticParticle();
    KSTrajMagneticParticle(const KSTrajMagneticParticle& aParticle);
    ~KSTrajMagneticParticle() override;

    //**********
    //assignment
    //**********

  public:
    void PullFrom(const KSParticle& aParticle);
    void PushTo(KSParticle& aParticle);

    KSTrajMagneticParticle& operator=(const double& anOperand);

    KSTrajMagneticParticle& operator=(const KSMathArray<5>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajMagneticParticle& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajMagneticParticle& operator=(const KSTrajMagneticParticle& anOperand);

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
    const KGeoBag::KThreeVector& GetPosition() const;
    const KGeoBag::KThreeVector& GetMomentum() const;
    const KGeoBag::KThreeVector& GetVelocity() const;
    const double& GetLorentzFactor() const;
    const double& GetKineticEnergy() const;

    const KGeoBag::KThreeVector& GetMagneticField() const;
    const KGeoBag::KThreeVector& GetElectricField() const;
    const double& GetElectricPotential() const;
    const KGeoBag::KThreeMatrix& GetMagneticGradient() const;

    const KGeoBag::KThreeVector& GetGuidingCenter() const;
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
    mutable KGeoBag::KThreeVector fPosition;
    mutable KGeoBag::KThreeVector fMomentum;
    mutable KGeoBag::KThreeVector fVelocity;
    mutable double fLorentzFactor;
    mutable double fKineticEnergy;

    mutable KGeoBag::KThreeVector fMagneticField;
    mutable KGeoBag::KThreeVector fElectricField;
    mutable KGeoBag::KThreeMatrix fMagneticGradient;
    mutable double fElectricPotential;

    mutable KGeoBag::KThreeVector fGuidingCenter;
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

    mutable void (KSTrajMagneticParticle::*fGetMagneticFieldPtr)() const;
    mutable void (KSTrajMagneticParticle::*fGetElectricFieldPtr)() const;
    mutable void (KSTrajMagneticParticle::*fGetMagneticGradientPtr)() const;
    mutable void (KSTrajMagneticParticle::*fGetElectricPotentialPtr)() const;
};

inline KSTrajMagneticParticle& KSTrajMagneticParticle::operator=(const double& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajMagneticParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajMagneticParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajMagneticParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajMagneticParticle::RecalculateElectricPotential;
    return *this;
}

inline KSTrajMagneticParticle& KSTrajMagneticParticle::operator=(const KSMathArray<5>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajMagneticParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajMagneticParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajMagneticParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajMagneticParticle::RecalculateElectricPotential;
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajMagneticParticle&
KSTrajMagneticParticle::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajMagneticParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajMagneticParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajMagneticParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajMagneticParticle::RecalculateElectricPotential;
    return *this;
}

inline KSTrajMagneticParticle& KSTrajMagneticParticle::operator=(const KSTrajMagneticParticle& aParticle)
{
    this->KSMathArray<5>::operator=(aParticle);

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

    return *this;
}

}  // namespace Kassiopeia

#endif

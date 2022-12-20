#ifndef Kassiopeia_KSTrajElectricParticle_h_
#define Kassiopeia_KSTrajElectricParticle_h_

#include "KSMathArray.h"
#include "KSParticle.h"

namespace Kassiopeia
{

class KSTrajElectricParticle : public KSMathArray<5>
{
  public:
    KSTrajElectricParticle();
    KSTrajElectricParticle(const KSTrajElectricParticle& aParticle);
    ~KSTrajElectricParticle() override;

    //**********
    //assignment
    //**********

  public:
    void PullFrom(const KSParticle& aParticle);
    void PushTo(KSParticle& aParticle);

    KSTrajElectricParticle& operator=(const double& anOperand);

    KSTrajElectricParticle& operator=(const KSMathArray<5>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajElectricParticle& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajElectricParticle& operator=(const KSTrajElectricParticle& anOperand);

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

    mutable void (KSTrajElectricParticle::*fGetMagneticFieldPtr)() const;
    mutable void (KSTrajElectricParticle::*fGetElectricFieldPtr)() const;
    mutable void (KSTrajElectricParticle::*fGetMagneticGradientPtr)() const;
    mutable void (KSTrajElectricParticle::*fGetElectricPotentialPtr)() const;
};

inline KSTrajElectricParticle& KSTrajElectricParticle::operator=(const double& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajElectricParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajElectricParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajElectricParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajElectricParticle::RecalculateElectricPotential;
    return *this;
}

inline KSTrajElectricParticle& KSTrajElectricParticle::operator=(const KSMathArray<5>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajElectricParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajElectricParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajElectricParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajElectricParticle::RecalculateElectricPotential;
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajElectricParticle&
KSTrajElectricParticle::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<5>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajElectricParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajElectricParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajElectricParticle::RecalculateMagneticGradient;
    fGetElectricPotentialPtr = &KSTrajElectricParticle::RecalculateElectricPotential;
    return *this;
}

inline KSTrajElectricParticle& KSTrajElectricParticle::operator=(const KSTrajElectricParticle& aParticle)
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

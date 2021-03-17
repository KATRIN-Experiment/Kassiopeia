#ifndef Kassiopeia_KSTrajAdiabaticParticle_h_
#define Kassiopeia_KSTrajAdiabaticParticle_h_

#include "KSMathArray.h"
#include "KSParticle.h"
#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

class KSTrajAdiabaticParticle : public KSMathArray<8>
{
  public:
    KSTrajAdiabaticParticle();
    KSTrajAdiabaticParticle(const KSTrajAdiabaticParticle& aParticle);
    ~KSTrajAdiabaticParticle() override;

    //**********
    //assignment
    //**********

  public:
    void PullFrom(const KSParticle& aParticle);
    void PushTo(KSParticle& aParticle);

    KSTrajAdiabaticParticle& operator=(const double& anOperand);

    KSTrajAdiabaticParticle& operator=(const KSMathArray<8>& anOperand);

    template<class XLeft, class XOperation, class XRight>
    KSTrajAdiabaticParticle& operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand);

    KSTrajAdiabaticParticle& operator=(const KSTrajAdiabaticParticle& aParticle);

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
    const double& GetTime() const;  //basic
    const double& GetLength() const;
    const KGeoBag::KThreeVector& GetPosition() const;
    const KGeoBag::KThreeVector& GetMomentum() const;
    const KGeoBag::KThreeVector& GetVelocity() const;
    const double& GetLorentzFactor() const;
    const double& GetKineticEnergy() const;

    const KGeoBag::KThreeVector& GetMagneticField() const;
    void SetMagneticField(const KGeoBag::KThreeVector& aField) const;
    const KGeoBag::KThreeVector& GetElectricField() const;
    const KGeoBag::KThreeMatrix& GetMagneticGradient() const;
    const std::pair<const KGeoBag::KThreeVector&, const KGeoBag::KThreeMatrix&> GetMagneticFieldAndGradient() const;
    const double& GetElectricPotential() const;
    const double& GetElectricPotentialRP() const;
    const std::pair<const KGeoBag::KThreeVector&, const double&> GetElectricFieldAndPotential() const;

    const KGeoBag::KThreeVector& GetGuidingCenter() const;  //basic
    const double& GetLongMomentum() const;                  //basic
    const double& GetTransMomentum() const;                 //basic
    const double& GetLongVelocity() const;
    const double& GetTransVelocity() const;
    const double& GetCyclotronFrequency() const;
    const double& GetOrbitalMagneticMoment() const;

    void SetAlpha(const KGeoBag::KThreeVector& anAlpha);
    const KGeoBag::KThreeVector& GetAlpha() const;

    void SetBeta(const KGeoBag::KThreeVector& aBeta);
    const KGeoBag::KThreeVector& GetBeta() const;

    void SetPhase(const double& aPhase);
    const double& GetPhase() const;


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
    mutable double fElectricPotentialRP;

    mutable KGeoBag::KThreeVector fGuidingCenter;
    mutable double fLongMomentum;
    mutable double fTransMomentum;
    mutable double fLongVelocity;
    mutable double fTransVelocity;
    mutable double fCyclotronFrequency;
    mutable double fOrbitalMagneticMoment;

    KGeoBag::KThreeVector fAlpha;
    KGeoBag::KThreeVector fBeta;
    double fLastTime;
    KGeoBag::KThreeVector fLastPosition;
    KGeoBag::KThreeVector fLastMomentum;
    mutable double fPhase;

    //*****
    //cache
    //*****

  private:
    void DoNothing() const;
    void RecalculateMagneticField() const;
    void RecalculateElectricField() const;
    void RecalculateMagneticGradient() const;
    void RecalculateMagneticFieldAndGradient() const;
    void RecalculateElectricPotential() const;
    void RecalculateElectricPotentialRP() const;
    void RecalculateElectricFieldAndPotential() const;

    mutable void (KSTrajAdiabaticParticle::*fGetMagneticFieldPtr)() const;
    mutable void (KSTrajAdiabaticParticle::*fGetElectricFieldPtr)() const;
    mutable void (KSTrajAdiabaticParticle::*fGetMagneticGradientPtr)() const;
    mutable void (KSTrajAdiabaticParticle::*fGetMagneticFieldAndGradientPtr)() const;
    mutable void (KSTrajAdiabaticParticle::*fGetElectricPotentialPtr)() const;
    mutable void (KSTrajAdiabaticParticle::*fGetElectricPotentialRPPtr)() const;
    mutable void (KSTrajAdiabaticParticle::*fGetElectricFieldAndPotentialPtr)() const;
};

inline KSTrajAdiabaticParticle& KSTrajAdiabaticParticle::operator=(const double& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajAdiabaticParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticGradient;
    fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticFieldAndGradient;
    fGetElectricPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotential;
    fGetElectricPotentialRPPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotentialRP;
    fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricFieldAndPotential;
    return *this;
}

inline KSTrajAdiabaticParticle& KSTrajAdiabaticParticle::operator=(const KSMathArray<8>& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajAdiabaticParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticGradient;
    fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticFieldAndGradient;
    fGetElectricPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotential;
    fGetElectricPotentialRPPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotentialRP;
    fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricFieldAndPotential;
    return *this;
}

template<class XLeft, class XOperation, class XRight>
inline KSTrajAdiabaticParticle&
KSTrajAdiabaticParticle::operator=(const KSMathExpression<XLeft, XOperation, XRight>& anOperand)
{
    this->KSMathArray<8>::operator=(anOperand);
    fGetMagneticFieldPtr = &KSTrajAdiabaticParticle::RecalculateMagneticField;
    fGetElectricFieldPtr = &KSTrajAdiabaticParticle::RecalculateElectricField;
    fGetMagneticGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticGradient;
    fGetMagneticFieldAndGradientPtr = &KSTrajAdiabaticParticle::RecalculateMagneticFieldAndGradient;
    fGetElectricPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotential;
    fGetElectricPotentialRPPtr = &KSTrajAdiabaticParticle::RecalculateElectricPotentialRP;
    fGetElectricFieldAndPotentialPtr = &KSTrajAdiabaticParticle::RecalculateElectricFieldAndPotential;
    return *this;
}

inline KSTrajAdiabaticParticle& KSTrajAdiabaticParticle::operator=(const KSTrajAdiabaticParticle& aParticle)
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
    fElectricPotentialRP = aParticle.fElectricPotentialRP;

    fGuidingCenter = aParticle.fGuidingCenter;
    fLongMomentum = aParticle.fLongMomentum;
    fTransMomentum = aParticle.fTransMomentum;
    fLongVelocity = aParticle.fLongVelocity;
    fTransVelocity = aParticle.fTransVelocity;
    fCyclotronFrequency = aParticle.fCyclotronFrequency;
    fOrbitalMagneticMoment = aParticle.fOrbitalMagneticMoment;

    fLastTime = aParticle.fLastTime;
    fLastPosition = aParticle.fLastPosition;
    fLastMomentum = aParticle.fLastMomentum;
    fAlpha = aParticle.fAlpha;
    fBeta = aParticle.fBeta;

    fGetMagneticFieldPtr = aParticle.fGetMagneticFieldPtr;
    fGetElectricFieldPtr = aParticle.fGetElectricFieldPtr;
    fGetMagneticGradientPtr = aParticle.fGetMagneticGradientPtr;
    fGetMagneticFieldAndGradientPtr = aParticle.fGetMagneticFieldAndGradientPtr;
    fGetElectricPotentialPtr = aParticle.fGetElectricPotentialPtr;
    fGetElectricPotentialRPPtr = aParticle.fGetElectricPotentialRPPtr;
    fGetElectricFieldAndPotentialPtr = aParticle.fGetElectricFieldAndPotentialPtr;

    return *this;
}

}  // namespace Kassiopeia

#endif

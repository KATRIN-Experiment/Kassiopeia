#ifndef Kassiopeia_KSTrajAdiabaticSpinParticle_h_
#define Kassiopeia_KSTrajAdiabaticSpinParticle_h_

#include "KSMathArray.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSTrajAdiabaticSpinParticle :
        public KSMathArray< 10 >
    {
        public:
            KSTrajAdiabaticSpinParticle();
            ~KSTrajAdiabaticSpinParticle();

            //**********
            //assignment
            //**********

        public:
            void PullFrom( const KSParticle& aParticle );
            void PushTo( KSParticle& aParticle ) const;

            KSTrajAdiabaticSpinParticle& operator=( const double& anOperand );

            KSTrajAdiabaticSpinParticle& operator=( const KSMathArray< 10 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajAdiabaticSpinParticle& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajAdiabaticSpinParticle& operator=( const KSTrajAdiabaticSpinParticle& anOperand );

            //***********
            //calculators
            //***********

        public:
            static void SetMagneticFieldCalculator( KSMagneticField* aMagneticField );
            static KSMagneticField* GetMagneticFieldCalculator();

            static void SetElectricFieldCalculator( KSElectricField* anElectricField );
            static KSElectricField* GetElectricFieldCalculator();

            //****************
            //static variables
            //****************

        public:
            static void SetMass( const double& aMass );
            static const double& GetMass();

            static void SetCharge( const double& aCharge );
            static const double& GetCharge();

            static void SetSpinMagnitude( const double& aSpinMagnitude );
            static const double& GetSpinMagnitude();

            static void SetGyromagneticRatio( const double& aGyromagneticRatio );
            static const double& GetGyromagneticRatio();

            void FixSpin();

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
            const double& GetSpinPrecessionFrequency() const;
            const double& GetOrbitalMagneticMoment() const;

            const double& GetAlignedSpin() const;
            const double& GetSpinAngle() const;

        private:
            static KSMagneticField* fMagneticFieldCalculator;
            static KSElectricField* fElectricFieldCalculator;

            static double fMass;
            static double fCharge;
            static double fSpinMagnitude;
            static double fGyromagneticRatio;

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
            mutable double fSpinPrecessionFrequency;
            mutable double fOrbitalMagneticMoment;

            mutable double fAlignedSpin;
            mutable double fSpinAngle;

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

            mutable void (KSTrajAdiabaticSpinParticle::*fGetMagneticFieldPtr)() const;
            mutable void (KSTrajAdiabaticSpinParticle::*fGetElectricFieldPtr)() const;
            mutable void (KSTrajAdiabaticSpinParticle::*fGetMagneticGradientPtr)() const;
            mutable void (KSTrajAdiabaticSpinParticle::*fGetElectricPotentialPtr)() const;
            mutable void (KSTrajAdiabaticSpinParticle::*fGetElectricGradientPtr)() const;
    };

    inline KSTrajAdiabaticSpinParticle& KSTrajAdiabaticSpinParticle::operator=( const double& anOperand )
    {
        this->KSMathArray< 10 >::operator =( anOperand );
        fGetMagneticFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricPotential;
        fGetElectricGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricGradient;
        return *this;
    }

    inline KSTrajAdiabaticSpinParticle& KSTrajAdiabaticSpinParticle::operator=( const KSMathArray< 10 >& anOperand )
    {
        this->KSMathArray< 10 >::operator =( anOperand );
        fGetMagneticFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricPotential;
        fGetElectricGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricGradient;
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajAdiabaticSpinParticle& KSTrajAdiabaticSpinParticle::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 10 >::operator =( anOperand );
        fGetMagneticFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricPotential;
        fGetElectricGradientPtr = &KSTrajAdiabaticSpinParticle::RecalculateElectricGradient;
        return *this;
    }

    inline KSTrajAdiabaticSpinParticle& KSTrajAdiabaticSpinParticle::operator=( const KSTrajAdiabaticSpinParticle& aParticle )
    {
        this->KSMathArray< 10 >::operator =( aParticle );

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

        fAlignedSpin = aParticle.fAlignedSpin;
        fSpinAngle = aParticle.fSpinAngle;

        fGetMagneticFieldPtr = aParticle.fGetMagneticFieldPtr;
        fGetElectricFieldPtr = aParticle.fGetElectricFieldPtr;
        fGetMagneticGradientPtr = aParticle.fGetMagneticGradientPtr;
        fGetElectricPotentialPtr = aParticle.fGetElectricPotentialPtr;
        fGetElectricGradientPtr = aParticle.fGetElectricGradientPtr;

        return *this;
    }

}

#endif

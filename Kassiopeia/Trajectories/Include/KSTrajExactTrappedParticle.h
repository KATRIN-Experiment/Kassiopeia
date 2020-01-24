#ifndef Kassiopeia_KSTrajExactTrappedParticle_h_
#define Kassiopeia_KSTrajExactTrappedParticle_h_

#include "KSMathArray.h"
#include "KSParticle.h"

namespace Kassiopeia
{

    class KSTrajExactTrappedParticle :
        public KSMathArray< 8 >
    {
        public:
            KSTrajExactTrappedParticle();
            KSTrajExactTrappedParticle( const KSTrajExactTrappedParticle& anOperand );
            ~KSTrajExactTrappedParticle();

            //**********
            //assignment
            //**********

        public:
            void PullFrom( const KSParticle& aParticle );
            void PushTo( KSParticle& aParticle ) const;

            KSTrajExactTrappedParticle& operator=( const double& anOperand );

            KSTrajExactTrappedParticle& operator=( const KSMathArray< 8 >& anOperand );

            template< class XLeft, class XOperation, class XRight >
            KSTrajExactTrappedParticle& operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand );

            KSTrajExactTrappedParticle& operator=( const KSTrajExactTrappedParticle& anOperand );

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

            mutable void (KSTrajExactTrappedParticle::*fGetMagneticFieldPtr)() const;
            mutable void (KSTrajExactTrappedParticle::*fGetElectricFieldPtr)() const;
            mutable void (KSTrajExactTrappedParticle::*fGetMagneticGradientPtr)() const;
            mutable void (KSTrajExactTrappedParticle::*fGetElectricPotentialPtr)() const;
            mutable void (KSTrajExactTrappedParticle::*fGetElectricGradientPtr)() const;
    };

    inline KSTrajExactTrappedParticle& KSTrajExactTrappedParticle::operator=( const double& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        fGetMagneticFieldPtr = &KSTrajExactTrappedParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajExactTrappedParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajExactTrappedParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajExactTrappedParticle::RecalculateElectricPotential;
        fGetElectricGradientPtr = &KSTrajExactTrappedParticle::RecalculateElectricGradient;
        return *this;
    }

    inline KSTrajExactTrappedParticle& KSTrajExactTrappedParticle::operator=( const KSMathArray< 8 >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        fGetMagneticFieldPtr = &KSTrajExactTrappedParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajExactTrappedParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajExactTrappedParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajExactTrappedParticle::RecalculateElectricPotential;
        fGetElectricGradientPtr = &KSTrajExactTrappedParticle::RecalculateElectricGradient;
        return *this;
    }

    template< class XLeft, class XOperation, class XRight >
    inline KSTrajExactTrappedParticle& KSTrajExactTrappedParticle::operator=( const KSMathExpression< XLeft, XOperation, XRight >& anOperand )
    {
        this->KSMathArray< 8 >::operator =( anOperand );
        fGetMagneticFieldPtr = &KSTrajExactTrappedParticle::RecalculateMagneticField;
        fGetElectricFieldPtr = &KSTrajExactTrappedParticle::RecalculateElectricField;
        fGetMagneticGradientPtr = &KSTrajExactTrappedParticle::RecalculateMagneticGradient;
        fGetElectricPotentialPtr = &KSTrajExactTrappedParticle::RecalculateElectricPotential;
        fGetElectricGradientPtr = &KSTrajExactTrappedParticle::RecalculateElectricGradient;
        return *this;
    }

    inline KSTrajExactTrappedParticle& KSTrajExactTrappedParticle::operator=( const KSTrajExactTrappedParticle& aParticle )
    {
        this->KSMathArray< 8 >::operator =( aParticle );

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

}

#endif

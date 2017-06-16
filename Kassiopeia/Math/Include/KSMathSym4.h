#ifndef Kassiopeia_KSMathSym4_h_
#define Kassiopeia_KSMathSym4_h_

#include <limits>
#include <cmath>

#include "KSMathIntegrator.h"

#define KSMATHSYM4_STAGE 3
//Symplectic Integration Method Outlined in 'Symplectic and energy-conserving algorithms for solving magnetic field trajectories' S. Chin, Phys Rev E 77, 066401 (2008)
//NOT Recommended for Use Outside ExactTrappedParticle

namespace Kassiopeia
{

    template< class XSystemType >
    class KSMathSym4 :
        public KSMathIntegrator< XSystemType >
    {
        public:
            typedef XSystemType SystemType;
            typedef KSMathDifferentiator< SystemType > DifferentiatorType;
            typedef typename SystemType::ValueType ValueType;
            typedef typename SystemType::DerivativeType DerivativeType;
            typedef typename SystemType::ErrorType ErrorType;

        public:
            KSMathSym4();
            virtual ~KSMathSym4();

        public:
            virtual void Integrate( double aTime,
                                    const DifferentiatorType& aTerm,
                                    const ValueType& anInitialValue,
                                    const double& aStep,
                                    ValueType& aFinalValue,
                                    ErrorType& anError ) const;
        private:
            enum
            {
                eSubsteps = 11, eConditions = eSubsteps - 1
            };

            mutable ValueType fValues[eConditions];
            mutable DerivativeType fDerivatives[eSubsteps];
            static const double fT[ KSMATHSYM4_STAGE ];
            static const double fV[ KSMATHSYM4_STAGE ];

    };

    template< class XSystemType >
    KSMathSym4< XSystemType >::KSMathSym4()
    {
    }

    template< class XSystemType >
    KSMathSym4< XSystemType >::~KSMathSym4()
    {
    }

    template< class XSystemType >
    void KSMathSym4< XSystemType >::Integrate(double /*aTime*/, const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue, ErrorType& /*anError*/ ) const
    {
        fValues[0] = anInitialValue;
        //
        //Only Update Position Values
        aTerm.Differentiate(0, fValues[0], fDerivatives[0] );
        fValues[1]= anInitialValue + fT[2]* aStep * fDerivatives[0];
    
        //Only Update Momentum Values
        aTerm.Differentiate(fV[2] * aStep, fValues[1], fDerivatives[1] );
        fValues[2] = fValues[1] + fV[2] * aStep*fDerivatives[1];

        //Only Update Position Values
        aTerm.Differentiate(0, fValues[2], fDerivatives[2] );
        fValues[3]= fValues[2] + fT[1]* aStep * fDerivatives[2];
    
        //Only Update Momentum Values
        aTerm.Differentiate(fV[1] * aStep, fValues[3], fDerivatives[3] );
        fValues[4] = fValues[3] + fV[1] * aStep*fDerivatives[3];

        //Only Update Position Values
        aTerm.Differentiate(0, fValues[4], fDerivatives[4] );
        fValues[5]= fValues[4] + fT[0]* aStep * fDerivatives[4];
        
        //Only Update Momentum Values
        aTerm.Differentiate(fV[1] * aStep, fValues[5], fDerivatives[5] );
        fValues[6] = fValues[5] + fV[1] * aStep*fDerivatives[5];

        //Only Update Position Values
        aTerm.Differentiate(0, fValues[6], fDerivatives[6] );
        fValues[7]= fValues[6] + fT[1]* aStep * fDerivatives[6];

        //Only Update Momentum Values
        aTerm.Differentiate(fV[2] * aStep, fValues[7], fDerivatives[7] );
        fValues[8] = fValues[7] + fV[2] * aStep*fDerivatives[7];

        //Only Update Position Values
        aTerm.Differentiate(0, fValues[8], fDerivatives[8] );
        fValues[9]= fValues[8] + fT[2]* aStep * fDerivatives[8];


        aFinalValue = fValues[9];

        return;
    }
    template< class XSystemType >
    const double KSMathSym4< XSystemType >::fT[3] =
    {1.260093847963958,-0.299186203904051,0.169139279922072};

    template< class XSystemType >
    const double KSMathSym4< XSystemType >::fV[3] =
    { 0., -1./22. , 6. / 11.};
}

#endif

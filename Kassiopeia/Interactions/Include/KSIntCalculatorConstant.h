#ifndef Kassiopeia_KSIntCalculatorConstant_h_
#define Kassiopeia_KSIntCalculatorConstant_h_

#include "KSIntCalculator.h"

#include "KField.h"

namespace Kassiopeia
{
    class KSIntCalculatorConstant :
        public KSComponentTemplate< KSIntCalculatorConstant, KSIntCalculator >
    {
        public:
            KSIntCalculatorConstant();
            KSIntCalculatorConstant( const KSIntCalculatorConstant& aCopy );
            KSIntCalculatorConstant* Clone() const;
            virtual ~KSIntCalculatorConstant();

        public:
            void CalculateCrossSection( const KSParticle& aParticle, double& aCrossSection );
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        public:
            K_SET_GET( double, CrossSection ); // m^2
    };

}

#endif

#ifndef Kassiopeia_KSIntSurfaceUCN_h_
#define Kassiopeia_KSIntSurfaceUCN_h_

#include "KSSurfaceInteraction.h"

#include "KField.h"
#include "KMathBracketingSolver.h"
using katrin::KMathBracketingSolver;

namespace Kassiopeia
{

    class KSStep;

    class KSIntSurfaceUCN :
        public KSComponentTemplate< KSIntSurfaceUCN, KSSurfaceInteraction >
    {
        public:
            KSIntSurfaceUCN();
            KSIntSurfaceUCN( const KSIntSurfaceUCN& aCopy );
            KSIntSurfaceUCN* Clone() const;
            virtual ~KSIntSurfaceUCN();

        public:
            void ExecuteInteraction( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );
            void ExecuteReflection( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );
            void ExecuteTransmission( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aSecondaries );

        public:
            K_SET_GET( double, Eta ) // eta value (related to reflection probability)
            K_SET_GET( double, Alpha) // alpha value (probability of spin sign flipping, so 1/2 of spin flip probability)
            K_SET_GET( double, RealOpticalPotential )
            K_SET_GET( double, CorrelationLength ) // of the roughness

        private:
            double fTanThetaIn;
            double fExpThetaCoef;

        protected:
            double ValueFunction( const double& aValue ) const;
            KMathBracketingSolver fSolver;

    };

}

#endif

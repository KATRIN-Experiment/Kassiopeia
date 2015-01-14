#ifndef Kassiopeia_KSTrajInterpolatorFast_h_
#define Kassiopeia_KSTrajInterpolatorFast_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajInterpolatorFast :
        public KSComponentTemplate< KSTrajInterpolatorFast >,
        public KSTrajExactInterpolator,
        public KSTrajAdiabaticInterpolator,
        public KSTrajMagneticInterpolator
    {
        public:
            KSTrajInterpolatorFast();
            KSTrajInterpolatorFast( const KSTrajInterpolatorFast& aCopy );
            KSTrajInterpolatorFast* Clone() const;
            virtual ~KSTrajInterpolatorFast();

        public:
            virtual void Interpolate( const KSTrajExactDifferentiator& /*aDifferentiator*/, const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const double& aTimeStep, KSTrajExactParticle& anIntermediateParticle ) const;
            virtual void Interpolate( const KSTrajAdiabaticDifferentiator& /*aDifferentiator*/, const KSTrajAdiabaticParticle& anInitial, const KSTrajAdiabaticParticle& aFinal, const double& aValue, KSTrajAdiabaticParticle& anIntermediate ) const;
            virtual void Interpolate( const KSTrajMagneticDifferentiator& /*aDifferentiator*/, const KSTrajMagneticParticle& anInitial, const KSTrajMagneticParticle& aFinal, const double& aValue, KSTrajMagneticParticle& anIntermediate ) const;
    };

}

#endif

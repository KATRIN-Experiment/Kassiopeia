#ifndef Kassiopeia_KSTrajTermConstantForcePropagation_h_
#define Kassiopeia_KSTrajTermConstantForcePropagation_h_

#include "KSComponentTemplate.h"
#include "KField.h"
#include "KSTrajExactTypes.h"

namespace Kassiopeia
{

    class KSTrajTermConstantForcePropagation :
        public KSComponentTemplate< KSTrajTermConstantForcePropagation >,
        public KSTrajExactDifferentiator
    {
        public:
            KSTrajTermConstantForcePropagation();
            KSTrajTermConstantForcePropagation( const KSTrajTermConstantForcePropagation& aCopy );
            KSTrajTermConstantForcePropagation* Clone() const;
            virtual ~KSTrajTermConstantForcePropagation();

        public:
            virtual void Differentiate(double /*aTime*/, const KSTrajExactParticle& aValue,
                                        KSTrajExactDerivative& aDerivative ) const;
            void SetForce( const KThreeVector& aForce );

        private:
            KThreeVector fForce;
    };

}

#endif // Kassiopeia_KSTrajTermConstantForcePropagation_h_

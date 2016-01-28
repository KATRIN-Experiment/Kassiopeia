#ifndef Kassiopeia_KSTrajTermPropagation_h_
#define Kassiopeia_KSTrajTermPropagation_h_

#include "KSComponentTemplate.h"
#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajTermPropagation :
        public KSComponentTemplate< KSTrajTermPropagation >,
        public KSTrajExactDifferentiator,
        public KSTrajAdiabaticDifferentiator,
        public KSTrajElectricDifferentiator,
        public KSTrajMagneticDifferentiator
    {
        public:
            KSTrajTermPropagation();
            KSTrajTermPropagation( const KSTrajTermPropagation& aCopy );
            KSTrajTermPropagation* Clone() const;
            virtual ~KSTrajTermPropagation();

        public:
            virtual void Differentiate(double /*aTime*/, const KSTrajExactParticle& aValue, KSTrajExactDerivative& aDerivative ) const;
            virtual void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aValue, KSTrajAdiabaticDerivative& aDerivative ) const;
            virtual void Differentiate(double /*aTime*/, const KSTrajMagneticParticle& aValue, KSTrajMagneticDerivative& aDerivative ) const;
            virtual void Differentiate(double /*aTime*/, const KSTrajElectricParticle& aValue, KSTrajElectricDerivative& aDerivative ) const;

        public:
            typedef enum
            {
                eBackward = -1,
                eForward = 1
            } Direction;

            void SetDirection( const Direction& anDirection );

        private:
            Direction fDirection;
    };

}

#endif

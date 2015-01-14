#ifndef Kassiopeia_KSTrajTermPropagation_h_
#define Kassiopeia_KSTrajTermPropagation_h_

#include "KSComponentTemplate.h"
#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajTermPropagation :
        public KSComponentTemplate< KSTrajTermPropagation >,
        public KSTrajExactDifferentiator,
        public KSTrajAdiabaticDifferentiator,
        public KSTrajMagneticDifferentiator
    {
        public:
            KSTrajTermPropagation();
            KSTrajTermPropagation( const KSTrajTermPropagation& aCopy );
            KSTrajTermPropagation* Clone() const;
            virtual ~KSTrajTermPropagation();

        public:
            virtual void Differentiate( const KSTrajExactParticle& aValue, KSTrajExactDerivative& aDerivative ) const;
            virtual void Differentiate( const KSTrajAdiabaticParticle& aValue, KSTrajAdiabaticDerivative& aDerivative ) const;
            virtual void Differentiate( const KSTrajMagneticParticle& aValue, KSTrajMagneticDerivative& aDerivative ) const;

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

#ifndef Kassiopeia_KSTrajTermGravity_h_
#define Kassiopeia_KSTrajTermGravity_h_

#include "KSComponentTemplate.h"
#include "KSTrajExactTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajTermGravity :
        public KSComponentTemplate< KSTrajTermGravity >,
        public KSTrajExactDifferentiator,
        public KSTrajExactSpinDifferentiator,
        // public KSTrajAdiabaticDifferentiator,
        public KSTrajAdiabaticSpinDifferentiator//,
        // public KSTrajElectricDifferentiator,
        // public KSTrajMagneticDifferentiator
    {
        public:
            KSTrajTermGravity();
            KSTrajTermGravity( const KSTrajTermGravity& aCopy );
            KSTrajTermGravity* Clone() const;
            virtual ~KSTrajTermGravity();

        public:
          virtual void Differentiate(double /*aTime*/, const KSTrajExactParticle& aValue, KSTrajExactDerivative& aDerivative ) const;
          virtual void Differentiate(double /*aTime*/, const KSTrajExactSpinParticle& aValue, KSTrajExactSpinDerivative& aDerivative ) const;
          // virtual void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aValue, KSTrajAdiabaticDerivative& aDerivative ) const;
          virtual void Differentiate(double /*aTime*/, const KSTrajAdiabaticSpinParticle& aValue, KSTrajAdiabaticSpinDerivative& aDerivative ) const;
          // virtual void Differentiate(double /*aTime*/, const KSTrajMagneticParticle& aValue, KSTrajMagneticDerivative& aDerivative ) const;
          // virtual void Differentiate(double /*aTime*/, const KSTrajElectricParticle& aValue, KSTrajElectricDerivative& aDerivative ) const;

        public:
            void SetGravity( const KThreeVector& aGravity );

        private:
            KThreeVector fGravity;

    };

}

#endif

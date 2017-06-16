#ifndef Kassiopeia_KSTrajTermSynchrotron_h_
#define Kassiopeia_KSTrajTermSynchrotron_h_

#include "KSComponentTemplate.h"
#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajExactTrappedTypes.h"

namespace Kassiopeia
{

    class KSTrajTermSynchrotron :
        public KSComponentTemplate< KSTrajTermSynchrotron >,
        public KSTrajExactDifferentiator,
        public KSTrajAdiabaticDifferentiator,
        public KSTrajExactTrappedDifferentiator
    {
        public:
            KSTrajTermSynchrotron();
            KSTrajTermSynchrotron( const KSTrajTermSynchrotron& aCopy );
            KSTrajTermSynchrotron* Clone() const;
            virtual ~KSTrajTermSynchrotron();

        public:
            virtual void Differentiate(double /*aTime*/, const KSTrajExactParticle& aParticle, KSTrajExactDerivative& aDerivative ) const;
            virtual void Differentiate(double /*aTime*/, const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const;
            virtual void Differentiate(double aTime, const KSTrajExactTrappedParticle& aParticle, KSTrajExactTrappedDerivative& aDerivative) const;

        public:
            void SetEnhancement( const double& anEnhancement );
            void SetOldMethode( const bool& aBool );

        private:
            double fEnhancement;
            bool fOldMethode;
    };

}

#endif

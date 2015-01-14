#ifndef Kassiopeia_KSTrajTermSynchrotron_h_
#define Kassiopeia_KSTrajTermSynchrotron_h_

#include "KSComponentTemplate.h"
#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajTermSynchrotron :
        public KSComponentTemplate< KSTrajTermSynchrotron >,
        public KSTrajExactDifferentiator,
        public KSTrajAdiabaticDifferentiator
    {
        public:
            KSTrajTermSynchrotron();
            KSTrajTermSynchrotron( const KSTrajTermSynchrotron& aCopy );
            KSTrajTermSynchrotron* Clone() const;
            virtual ~KSTrajTermSynchrotron();

        public:
            virtual void Differentiate( const KSTrajExactParticle& aParticle, KSTrajExactDerivative& aDerivative ) const;
            virtual void Differentiate( const KSTrajAdiabaticParticle& aParticle, KSTrajAdiabaticDerivative& aDerivative ) const;

        public:
            void SetEnhancement( const double& anEnhancement );

        private:
            double fEnhancement;
    };

}

#endif

#ifndef Kassiopeia_KSTrajControlCyclotron_h_
#define Kassiopeia_KSTrajControlCyclotron_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlCyclotron :
        public KSComponentTemplate< KSTrajControlCyclotron >,
        public KSTrajExactControl,
        public KSTrajAdiabaticControl
    {
        public:
            KSTrajControlCyclotron();
            KSTrajControlCyclotron( const KSTrajControlCyclotron& aCopy );
            KSTrajControlCyclotron* Clone() const;
            virtual ~KSTrajControlCyclotron();

        public:
            void Calculate( const KSTrajExactParticle& aParticle, double& aValue );
            void Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError& anError, bool& aFlag );

        public:
            void SetFraction( const double& aFraction );

        private:
            double fFraction;
    };

    inline void KSTrajControlCyclotron::SetFraction( const double& aFraction )
    {
        fFraction = aFraction;
        return;
    }

}

#endif


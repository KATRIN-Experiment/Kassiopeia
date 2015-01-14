#ifndef Kassiopeia_KSTrajControlLength_h_
#define Kassiopeia_KSTrajControlLength_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlLength :
        public KSComponentTemplate< KSTrajControlLength >,
        public KSTrajExactControl,
        public KSTrajAdiabaticControl
    {
        public:
            KSTrajControlLength();KSTrajControlLength( const KSTrajControlLength& aCopy );
            KSTrajControlLength* Clone() const;virtual ~KSTrajControlLength();

        public:
            void Calculate( const KSTrajExactParticle& aParticle, double& aValue );
            void Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError& anError, bool& aFlag );

        public:
            void SetLength( const double& aLength );

        private:
            double fLength;
    };

    inline void KSTrajControlLength::SetLength( const double& aLength )
    {
        fLength = aLength;
        return;
    }

}

#endif

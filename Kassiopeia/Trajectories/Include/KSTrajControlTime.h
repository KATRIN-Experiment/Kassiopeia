#ifndef Kassiopeia_KSTrajControlTime_h_
#define Kassiopeia_KSTrajControlTime_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"
#include "KSTrajElectricTypes.h"
#include "KSTrajMagneticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlTime :
        public KSComponentTemplate< KSTrajControlTime >,
        public KSTrajExactControl,
        public KSTrajAdiabaticControl,
		public KSTrajElectricControl,
        public KSTrajMagneticControl
    {
        public:
            KSTrajControlTime();KSTrajControlTime( const KSTrajControlTime& aCopy );
            KSTrajControlTime* Clone() const;virtual ~KSTrajControlTime();

        public:
            void Calculate( const KSTrajExactParticle& aParticle, double& aValue );
            void Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError& anError, bool& aFlag );

            void Calculate( const KSTrajElectricParticle& aParticle, double& aValue );
            void Check( const KSTrajElectricParticle& anInitialParticle, const KSTrajElectricParticle& aFinalParticle, const KSTrajElectricError& anError, bool& aFlag );

            void Calculate( const KSTrajMagneticParticle& aParticle, double& aValue );
            void Check( const KSTrajMagneticParticle& anInitialParticle, const KSTrajMagneticParticle& aFinalParticle, const KSTrajMagneticError& anError, bool& aFlag );

        public:
            void SetTime( const double& aTime );

        private:
            double fTime;
    };

    inline void KSTrajControlTime::SetTime( const double& aTime )
    {
        fTime = aTime;
        return;
    }

}

#endif


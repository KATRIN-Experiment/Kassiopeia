#ifndef Kassiopeia_KSTrajControlEnergy_h_
#define Kassiopeia_KSTrajControlEnergy_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlEnergy :
        public KSComponentTemplate< KSTrajControlEnergy >,
        public KSTrajExactControl,
        public KSTrajAdiabaticControl
    {
        public:
            KSTrajControlEnergy();KSTrajControlEnergy( const KSTrajControlEnergy& aCopy );
            KSTrajControlEnergy* Clone() const;virtual ~KSTrajControlEnergy();

        public:
            void Calculate( const KSTrajExactParticle& aParticle, double& aValue );
            void Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError& anError, bool& aFlag );

        public:
            void SetLowerLimit( const double& aLowerLimit );
            void SetUpperLimit( const double& aUpperLimit );

        protected:
            virtual void ActivateObject();

        private:
            double fLowerLimit;
            double fUpperLimit;
            double fTimeStep;
            bool fFirstStep;
    };

    inline void KSTrajControlEnergy::SetLowerLimit( const double& aLowerLimit )
    {
        fLowerLimit = aLowerLimit;
        return;
    }
    inline void KSTrajControlEnergy::SetUpperLimit( const double& aUpperLimit )
    {
        fUpperLimit = aUpperLimit;
        return;
    }

}

#endif

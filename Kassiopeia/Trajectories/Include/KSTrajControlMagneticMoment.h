#ifndef Kassiopeia_KSTrajControlMagneticMoment_h_
#define Kassiopeia_KSTrajControlMagneticMoment_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlMagneticMoment :
        public KSComponentTemplate< KSTrajControlMagneticMoment >,
        public KSTrajExactControl,
        public KSTrajAdiabaticControl
    {
        public:
            KSTrajControlMagneticMoment();
            KSTrajControlMagneticMoment( const KSTrajControlMagneticMoment& aCopy );
            KSTrajControlMagneticMoment* Clone() const;virtual ~KSTrajControlMagneticMoment();

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

    inline void KSTrajControlMagneticMoment::SetLowerLimit( const double& aLowerLimit )
    {
        fLowerLimit = aLowerLimit;
        return;
    }
    inline void KSTrajControlMagneticMoment::SetUpperLimit( const double& aUpperLimit )
    {
        fUpperLimit = aUpperLimit;
        return;
    }

}

#endif

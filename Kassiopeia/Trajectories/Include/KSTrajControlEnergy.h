#ifndef Kassiopeia_KSTrajControlEnergy_h_
#define Kassiopeia_KSTrajControlEnergy_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlEnergy :
        public KSComponentTemplate< KSTrajControlEnergy >,
        public KSTrajExactControl,
        public KSTrajExactSpinControl,
        public KSTrajAdiabaticSpinControl,
        public KSTrajAdiabaticControl
    {
        public:
            KSTrajControlEnergy();KSTrajControlEnergy( const KSTrajControlEnergy& aCopy );
            KSTrajControlEnergy* Clone() const;virtual ~KSTrajControlEnergy();

        public:
            void Calculate( const KSTrajExactParticle& aParticle, double& aValue );
            void Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError& anError, bool& aFlag );

            void Calculate( const KSTrajExactSpinParticle& aParticle, double& aValue );
            void Check( const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle, const KSTrajExactSpinError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticSpinParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle, const KSTrajAdiabaticSpinError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError& anError, bool& aFlag );

        public:
            void SetLowerLimit( const double& aLimit );
            void SetUpperLimit( const double& aLimit );
            void SetMinLength( const double& aLength );
            void SetMaxLength( const double& aLength );
            void SetInitialStep( const double& aTime );
            void SetAdjustmentFactorUp( const double& aFactor );
            void SetAdjustmentFactorDown( const double& aFactor );
            void SetStepRescale( const double& aFactor );

        protected:
            virtual void ActivateObject();

        private:
            double fLowerLimit;
            double fUpperLimit;
            double fMinLength;
            double fMaxLength;
            double fInitialStep;
            double fAdjustmentFactorUp;
            double fAdjustmentFactorDown;
            double fStepRescale;
            double fTimeStep;
            bool fFirstStep;
    };

    inline void KSTrajControlEnergy::SetLowerLimit( const double& aLimit )
    {
        if ( aLimit >= 0. )
            fLowerLimit = aLimit;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid lower limit <" << aLimit << ">" << eom;
        return;
    }
    inline void KSTrajControlEnergy::SetUpperLimit( const double& aLimit )
    {
        if ( aLimit <= 1. )
            fUpperLimit = aLimit;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid upper limit <" << aLimit << ">" << eom;
        return;
    }
    inline void KSTrajControlEnergy::SetMinLength( const double& aLength )
    {
        if ( aLength > 0. )
            fMinLength = aLength;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid minimal length <" << aLength << ">" << eom;
        return;
    }
    inline void KSTrajControlEnergy::SetMaxLength( const double& aLength )
    {
        if ( aLength > 0. )
            fMaxLength = aLength;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid maximal length <" << aLength << ">" << eom;
        return;
    }
    inline void KSTrajControlEnergy::SetInitialStep( const double& aTime )
    {
        if ( aTime > 0. )
            fInitialStep = aTime;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid initial step <" << aTime << ">" << eom;
        return;
    }
    inline void KSTrajControlEnergy::SetAdjustmentFactorUp( const double& aFactor )
    {
        if ( aFactor > 0. && aFactor < 1. )
            fAdjustmentFactorUp = aFactor;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid upwards adjustment factor <" << aFactor << ">" << eom;
        return;
    }
    inline void KSTrajControlEnergy::SetAdjustmentFactorDown( const double& aFactor )
    {
        if ( aFactor > 0. && aFactor < 1. )
            fAdjustmentFactorDown = aFactor;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid downwards adjustment factor <" << aFactor << ">" << eom;
        return;
    }
    inline void KSTrajControlEnergy::SetStepRescale( const double& aFactor )
    {
        if ( aFactor != 0. )
            fStepRescale = aFactor;
        else
            trajmsg( eWarning ) << "stepsize energy ignoring invalid step rescale factor <" << aFactor << ">" << eom;
        return;
    }
}

#endif

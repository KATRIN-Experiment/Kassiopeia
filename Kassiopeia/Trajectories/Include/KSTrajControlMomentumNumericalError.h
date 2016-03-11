#ifndef Kassiopeia_KSTrajControlMomentumNumericalError_h_
#define Kassiopeia_KSTrajControlMomentumNumericalError_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlMomentumNumericalError :
        public KSComponentTemplate< KSTrajControlMomentumNumericalError >,
        public KSTrajExactControl,
        public KSTrajAdiabaticControl
    {
        public:
            KSTrajControlMomentumNumericalError();KSTrajControlMomentumNumericalError( const KSTrajControlMomentumNumericalError& aCopy );
            KSTrajControlMomentumNumericalError* Clone() const;virtual ~KSTrajControlMomentumNumericalError();

        public:

            void Calculate( const KSTrajExactParticle& aParticle, double& aValue );
            void Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError& anError, bool& aFlag );


        protected:
            virtual void ActivateObject();

        public:

            void SetAbsoluteMomentumError(double error ){fAbsoluteError = error;};
            void SetSafetyFactor(double safety){fSafetyFactor = safety;};
            void SetSolverOrder(double order){fSolverOrder = order;};

        private:

            bool UpdateTimeStep(double error);

            static double fEpsilon;

            double fAbsoluteError; //max allowable error on momentum magnitude per step
            double fSafetyFactor; //safety factor for increasing/decreasing step size
            double fSolverOrder; //order of the associated runge-kutta stepper

            double fTimeStep;
            bool fFirstStep;
    };

}

#endif

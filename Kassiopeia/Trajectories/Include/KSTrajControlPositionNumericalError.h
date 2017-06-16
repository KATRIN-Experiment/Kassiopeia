#ifndef Kassiopeia_KSTrajControlPositionNumericalError_h_
#define Kassiopeia_KSTrajControlPositionNumericalError_h_

#include "KSComponentTemplate.h"

#include "KSTrajExactTypes.h"
#include "KSTrajExactSpinTypes.h"
#include "KSTrajAdiabaticSpinTypes.h"
#include "KSTrajAdiabaticTypes.h"

namespace Kassiopeia
{

    class KSTrajControlPositionNumericalError :
        public KSComponentTemplate< KSTrajControlPositionNumericalError >,
        public KSTrajExactControl,
        public KSTrajExactSpinControl,
        public KSTrajAdiabaticSpinControl,
        public KSTrajAdiabaticControl
    {
        public:
            KSTrajControlPositionNumericalError();KSTrajControlPositionNumericalError( const KSTrajControlPositionNumericalError& aCopy );
            KSTrajControlPositionNumericalError* Clone() const;virtual ~KSTrajControlPositionNumericalError();

        public:

            void Calculate( const KSTrajExactParticle& aParticle, double& aValue );
            void Check( const KSTrajExactParticle& anInitialParticle, const KSTrajExactParticle& aFinalParticle, const KSTrajExactError& anError, bool& aFlag );

            void Calculate( const KSTrajExactSpinParticle& aParticle, double& aValue );
            void Check( const KSTrajExactSpinParticle& anInitialParticle, const KSTrajExactSpinParticle& aFinalParticle, const KSTrajExactSpinError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticSpinParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticSpinParticle& anInitialParticle, const KSTrajAdiabaticSpinParticle& aFinalParticle, const KSTrajAdiabaticSpinError& anError, bool& aFlag );

            void Calculate( const KSTrajAdiabaticParticle& aParticle, double& aValue );
            void Check( const KSTrajAdiabaticParticle& anInitialParticle, const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError& anError, bool& aFlag );


        protected:
            virtual void ActivateObject();

        public:

            void SetAbsolutePositionError(double error ){fAbsoluteError = error;};
            void SetSafetyFactor(double safety){fSafetyFactor = safety;};
            void SetSolverOrder(double order){fSolverOrder = order;};

        private:

            bool UpdateTimeStep(double error);

            static double fEpsilon;

            double fAbsoluteError; //max allowable error on position magnitude per step
            double fSafetyFactor; //safety factor for increasing/decreasing step size
            double fSolverOrder; //order of the associated runge-kutta stepper

            double fTimeStep;
            bool fFirstStep;
    };

}

#endif

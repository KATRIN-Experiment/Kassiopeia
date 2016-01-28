#ifndef Kassiopeia_KSIntDecayCalculatorGlukhovIonisation_h_
#define Kassiopeia_KSIntDecayCalculatorGlukhovIonisation_h_

#include "KSIntDecayCalculator.h"
#include "KField.h"

namespace Kassiopeia
{
    class KSIntDecayCalculatorGlukhovIonisation :
        public KSComponentTemplate< KSIntDecayCalculatorGlukhovIonisation, KSIntDecayCalculator >
    {
        public:
            KSIntDecayCalculatorGlukhovIonisation();
            KSIntDecayCalculatorGlukhovIonisation( const KSIntDecayCalculatorGlukhovIonisation& aCopy );
            KSIntDecayCalculatorGlukhovIonisation* Clone() const;
            virtual ~KSIntDecayCalculatorGlukhovIonisation();

        public:
            void CalculateLifeTime( const KSParticle& aParticle, double& aLifeTime );
            void ExecuteInteraction( const KSParticle& anInitialParticle,
                                     KSParticle& aFinalParticle,
                                     KSParticleQueue& aSecondaries );


        public:
            K_SET_GET( long long, TargetPID )
            K_SET_GET( long long, minPID )
            K_SET_GET( long long, maxPID )
            K_SET_GET( double, Temperature )

            public:
                void SetDecayProductGenerator( KSGenerator* const aGenerator );
                KSGenerator* GetDecayProductGenerator() const;

            protected:
                KSGenerator* fDecayProductGenerator;

            private:
                static const double low_n_rates[32];
                static const double p_coefficients[3][4];
                double CalculateSpontaneousDecayRate(int n, int l);

                static const double b_ion[3][3][3];
                double a_ion(int l, int i, double T);
                static const double q_0,A,B;
                double q(double T);
                double tau_tilde(double T);
                double y(int n,double T);
                double CalculateRelativeIonisationRate(int n,int l,double T);


    };


}

#endif

//
// Created by trost on 03.06.15.
//

#ifndef KASPER_KSINTDECAYCALCULATORFERENCIONISATION_H
#define KASPER_KSINTDECAYCALCULATORFERENCIONISATION_H


#include "KSIntDecayCalculator.h"
#include "KField.h"
#include "RydbergFerenc.h"


namespace Kassiopeia {

    class KSIntDecayCalculatorFerencIonisation :
            public KSComponentTemplate<KSIntDecayCalculatorFerencIonisation, KSIntDecayCalculator> {

        public:
            KSIntDecayCalculatorFerencIonisation();
            virtual ~KSIntDecayCalculatorFerencIonisation();

            KSIntDecayCalculatorFerencIonisation(const KSIntDecayCalculatorFerencIonisation &aCopy);

            KSIntDecayCalculatorFerencIonisation *Clone() const;

        public:
            void CalculateLifeTime(const KSParticle &aParticle, double &aLifeTime);

            void ExecuteInteraction(const KSParticle &anInitialParticle,
                                          KSParticle &aFinalParticle,
                                          KSParticleQueue &aSecondaries);

        protected:
            virtual void InitializeComponent();


        public:
            K_SET_GET(long long, TargetPID)
            K_SET_GET(long long, minPID)
            K_SET_GET(long long, maxPID)
            K_SET_GET(double, Temperature)

        public:
            void SetDecayProductGenerator(KSGenerator *const aGenerator);

            KSGenerator *GetDecayProductGenerator() const;

        protected:
            KSGenerator *fDecayProductGenerator;

        private:
            double low_n_lifetimes[150][150];

        private:
            RydbergCalculator* fCalculator;



    };
}


#endif //KASPER_KSINTDECAYCALCULATORFERENCIONISATION_H
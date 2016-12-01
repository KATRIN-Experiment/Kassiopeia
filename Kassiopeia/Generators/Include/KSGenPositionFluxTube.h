#ifndef KSC_KSGenPositionFluxTube_h_
#define KSC_KSGenPositionFluxTube_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"
#include "KField.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

    class KSGenPositionFluxTube :
        public KSComponentTemplate< KSGenPositionFluxTube, KSGenCreator >
    {
        public:
    		KSGenPositionFluxTube();
    		KSGenPositionFluxTube( const KSGenPositionFluxTube& aCopy );
    		KSGenPositionFluxTube* Clone() const;
            virtual ~KSGenPositionFluxTube();

        public:
            virtual void Dice( KSParticleQueue* aPrimaryList );

        public:
            void SetPhiValue( KSGenValue* aPhiValue );
            void ClearPhiValue( KSGenValue* aPhiValue );

            void SetZValue( KSGenValue* anZValue );
            void ClearZValue( KSGenValue* anZValue );

            void AddMagneticField( KSMagneticField* aField );

        private:
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );


        private:
            KSGenValue* fPhiValue;
            KSGenValue* fZValue;
            std::vector<KSMagneticField*> fMagneticFields;
            ;K_SET( double, Flux );
            ;K_SET( int, NIntegrationSteps );
            ;K_SET( bool, OnlySurface );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();
    };

}

#endif

#ifndef KSC_KSGenPositionHomogeneousFluxTube_h_
#define KSC_KSGenPositionHomogeneousFluxTube_h_

#include "KSGenCreator.h"
#include "KSGenValue.h"
#include "KField.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

    class KSGenPositionHomogeneousFluxTube :
        public KSComponentTemplate< KSGenPositionHomogeneousFluxTube, KSGenCreator >
    {
        public:
    		KSGenPositionHomogeneousFluxTube();
    		KSGenPositionHomogeneousFluxTube( const KSGenPositionHomogeneousFluxTube& aCopy );
    		KSGenPositionHomogeneousFluxTube* Clone() const;
            virtual ~KSGenPositionHomogeneousFluxTube();

        public:
            virtual void Dice( KSParticleQueue* aPrimaryList );

        public:

            void AddMagneticField( KSMagneticField* aField );

        private:
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );


        private:
            std::vector<KSMagneticField*> fMagneticFields;
            ;K_SET( double, Flux );
            ;K_SET( double, Rmax );
            ;K_SET( int, NIntegrationSteps );
            ;K_SET( double, Zmin );
            ;K_SET( double, Zmax );
            ;K_SET( double, Phimin );
            ;K_SET( double, Phimax );


        protected:
            void InitializeComponent();
            void DeinitializeComponent();
    };

}

#endif

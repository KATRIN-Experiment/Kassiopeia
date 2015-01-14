#ifndef Kassiopeia_KSFieldElectricQuadrupole_h_
#define Kassiopeia_KSFieldElectricQuadrupole_h_

#include "KSElectricField.h"

namespace Kassiopeia
{

    class KSFieldElectricQuadrupole :
        public KSComponentTemplate< KSFieldElectricQuadrupole, KSElectricField >
    {
        public:
            KSFieldElectricQuadrupole();
            KSFieldElectricQuadrupole( const KSFieldElectricQuadrupole& aCopy );
            KSFieldElectricQuadrupole* Clone() const;
            virtual ~KSFieldElectricQuadrupole();

        public:
            void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aPotential );
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );

        public:
            void SetLocation( const KThreeVector& aLocation );
            void SetStrength( const double& aStrength );
            void SetLength( const double& aLength );
            void SetRadius( const double& aRadius );

        private:
            KThreeVector fLocation;
            double fStrength;
            double fLength;
            double fRadius;
            double fCharacteristic;
    };

}

#endif

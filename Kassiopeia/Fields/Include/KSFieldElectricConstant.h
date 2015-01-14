#ifndef Kassiopeia_KSFieldElectricConstant_h_
#define Kassiopeia_KSFieldElectricConstant_h_

#include "KSElectricField.h"

namespace Kassiopeia
{

    class KSFieldElectricConstant :
        public KSComponentTemplate< KSFieldElectricConstant, KSElectricField >
    {
        public:
            KSFieldElectricConstant();
            KSFieldElectricConstant( const KSFieldElectricConstant& aCopy );
            KSFieldElectricConstant* Clone() const;
            virtual ~KSFieldElectricConstant();

        public:
            void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aPotential );
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );

        public:
            void SetField( const KThreeVector& aFieldVector );

        private:
            KThreeVector fFieldVector;
    };

}

#endif

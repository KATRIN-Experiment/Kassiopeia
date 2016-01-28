#ifndef Kassiopeia_KSRootElectricField_h_
#define Kassiopeia_KSRootElectricField_h_

#include "KSElectricField.h"

#include "KSList.h"

namespace Kassiopeia
{

    class KSRootElectricField :
        public KSComponentTemplate< KSRootElectricField, KSElectricField >
    {
        public:
            KSRootElectricField();
            KSRootElectricField( const KSRootElectricField& aCopy );
            KSRootElectricField* Clone() const;
            virtual ~KSRootElectricField();

        public:
            virtual void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aPotential );
            virtual void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            virtual void CalculateFieldAndPotential( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField, double& aPotentia );

        public:
            void AddElectricField( KSElectricField* anElectricField );
            void RemoveElectricField( KSElectricField* anElectricField );

        private:
            double fCurrentPotential;
            KThreeVector fCurrentField;

            KSList< KSElectricField > fElectricFields;
    };

}

#endif

#ifndef Kassiopeia_KSRootMagneticField_h_
#define Kassiopeia_KSRootMagneticField_h_

#include "KSMagneticField.h"

#include "KSList.h"

namespace Kassiopeia
{

    class KSRootMagneticField :
        public KSComponentTemplate< KSRootMagneticField, KSMagneticField >
    {
        public:
            KSRootMagneticField();
            KSRootMagneticField( const KSRootMagneticField& aCopy );
            KSRootMagneticField* Clone() const;
            virtual ~KSRootMagneticField();

        public:
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient );
            void CalculateFieldAndGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField, KThreeMatrix& aGradient );

        public:
            void AddMagneticField( KSMagneticField* aMagneticField );
            void RemoveMagneticField( KSMagneticField* aMagneticField );

        private:
            KThreeVector fCurrentField;
            KThreeMatrix fCurrentGradient;

            KSList< KSMagneticField > fMagneticFields;
    };

}

#endif

#ifndef Kassiopeia_KSFieldMagneticConstant_h_
#define Kassiopeia_KSFieldMagneticConstant_h_

#include "KSMagneticField.h"

namespace Kassiopeia
{

    class KSFieldMagneticConstant :
        public KSComponentTemplate< KSFieldMagneticConstant, KSMagneticField >
    {
        public:
            KSFieldMagneticConstant();
            KSFieldMagneticConstant( const KSFieldMagneticConstant& aCopy );
            KSFieldMagneticConstant* Clone() const;
            virtual ~KSFieldMagneticConstant();

        public:
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient );

        public:
            void SetField( const KThreeVector& aFieldVector );

        private:
            KThreeVector fFieldVector;
    };

}

#endif

#ifndef Kassiopeia_KSFieldMagneticDipole_h_
#define Kassiopeia_KSFieldMagneticDipole_h_

#include "KSMagneticField.h"

namespace Kassiopeia
{

    class KSFieldMagneticDipole :
    public KSComponentTemplate< KSFieldMagneticDipole, KSMagneticField >
    {
        public:
            KSFieldMagneticDipole();
            KSFieldMagneticDipole( const KSFieldMagneticDipole& aCopy );
            KSFieldMagneticDipole* Clone() const;
            virtual ~KSFieldMagneticDipole();

        public:
            void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField );
            void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient );

        public:
            void SetLocation( const KThreeVector& aLocation );
            void SetMoment( const KThreeVector& aMoment );

        private:
            KThreeVector fLocation;
            KThreeVector fMoment;
    };

}

#endif

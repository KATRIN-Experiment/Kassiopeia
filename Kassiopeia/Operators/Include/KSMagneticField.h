#ifndef Kassiopeia_KSMagneticField_h_
#define Kassiopeia_KSMagneticField_h_

#include "KSComponentTemplate.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KThreeMatrix.hh"
using KGeoBag::KThreeMatrix;

namespace Kassiopeia
{

    class KSMagneticField :
        public KSComponentTemplate< KSMagneticField >
    {
        public:
            KSMagneticField();
            virtual ~KSMagneticField();

        public:
            virtual void CalculatePotential( const KThreeVector& /*aSamplePoint*/, const double& /*aSampleTime*/, KThreeVector& /*aPotential*/ ) {}
            virtual void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aField ) = 0;
            virtual void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aGradient ) = 0;
    };

}

#endif

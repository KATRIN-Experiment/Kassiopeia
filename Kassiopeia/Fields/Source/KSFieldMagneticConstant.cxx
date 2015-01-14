#include "KSFieldMagneticConstant.h"

namespace Kassiopeia
{

    KSFieldMagneticConstant::KSFieldMagneticConstant() :
        fFieldVector( 0., 0., 0. )
    {
    }
    KSFieldMagneticConstant::KSFieldMagneticConstant( const KSFieldMagneticConstant& aCopy ) :
        fFieldVector( aCopy.fFieldVector )
    {
    }
    KSFieldMagneticConstant* KSFieldMagneticConstant::Clone() const
    {
        return new KSFieldMagneticConstant( *this );
    }
    KSFieldMagneticConstant::~KSFieldMagneticConstant()
    {
    }

    void KSFieldMagneticConstant::CalculateField( const KThreeVector& /*aSamplePoint*/, const double& /*aSampleTime*/, KThreeVector& aField )
    {
        aField = fFieldVector;
        return;
    }

    void KSFieldMagneticConstant::CalculateGradient( const KThreeVector& /*aSamplePoint*/, const double& /*aSampleTime*/, KThreeMatrix& aGradient )
    {
        aGradient = KThreeMatrix::sZero;
        return;
    }

    void KSFieldMagneticConstant::SetField( const KThreeVector& aFieldVector )
    {
        fFieldVector = aFieldVector;
        return;
    }

}

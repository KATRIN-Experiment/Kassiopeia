#include "KSFieldElectricConstant.h"

namespace Kassiopeia
{

    KSFieldElectricConstant::KSFieldElectricConstant() :
        fFieldVector( 0., 0., 0. )
    {
    }
    KSFieldElectricConstant::KSFieldElectricConstant( const KSFieldElectricConstant& aCopy ) :
        fFieldVector( aCopy.fFieldVector )
    {
    }
    KSFieldElectricConstant* KSFieldElectricConstant::Clone() const
    {
        return new KSFieldElectricConstant( *this );
    }
    KSFieldElectricConstant::~KSFieldElectricConstant()
    {
    }

    void KSFieldElectricConstant::CalculatePotential( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, double& aPotential )
    {
        aPotential = fFieldVector.Dot( aSamplePoint );
        return;
    }

    void KSFieldElectricConstant::CalculateField( const KThreeVector& /*aSamplePoint*/, const double& /*aSampleTime*/, KThreeVector& aField )
    {
        aField = fFieldVector;
        return;
    }

    void KSFieldElectricConstant::SetField( const KThreeVector& aFieldVector )
    {
        fFieldVector = aFieldVector;
        return;
    }

}



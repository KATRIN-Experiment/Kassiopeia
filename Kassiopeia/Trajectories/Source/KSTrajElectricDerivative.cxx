#include "KSTrajElectricDerivative.h"

namespace Kassiopeia
{

    KSTrajElectricDerivative::KSTrajElectricDerivative()
    {
    }
    KSTrajElectricDerivative::KSTrajElectricDerivative( const KSTrajElectricDerivative& anOperand ) :
        KSMathArray<5>( anOperand )
    {
    }
    KSTrajElectricDerivative::~KSTrajElectricDerivative()
    {
    }

    void KSTrajElectricDerivative::AddToTime( const double& aTime )
    {
        fData[ 0 ] = aTime;
        return;
    }
    void KSTrajElectricDerivative::AddToSpeed( const double& aSpeed )
    {
        fData[ 1 ] = aSpeed;
        return;
    }
    void KSTrajElectricDerivative::AddToVelocity( const KThreeVector& aVelocity )
    {
        fData[ 2 ] = aVelocity.X();
        fData[ 3 ] = aVelocity.Y();
        fData[ 4 ] = aVelocity.Z();
        return;
    }


}

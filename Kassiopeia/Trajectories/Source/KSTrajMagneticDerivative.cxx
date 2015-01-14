#include "KSTrajMagneticDerivative.h"

namespace Kassiopeia
{

    KSTrajMagneticDerivative::KSTrajMagneticDerivative() :
        fDirectionSign( 1 )
    {
    }
    KSTrajMagneticDerivative::~KSTrajMagneticDerivative()
    {
    }

    void KSTrajMagneticDerivative::AddToTime( const double& aTime )
    {
        fData[ 0 ] = fDirectionSign * aTime;
        return;
    }
    void KSTrajMagneticDerivative::AddToSpeed( const double& aSpeed )
    {
        fData[ 1 ] = fDirectionSign * aSpeed;
        return;
    }
    void KSTrajMagneticDerivative::AddToVelocity( const KThreeVector& aVelocity )
    {
        fData[ 2 ] = fDirectionSign * aVelocity.X();
        fData[ 3 ] = fDirectionSign * aVelocity.Y();
        fData[ 4 ] = fDirectionSign * aVelocity.Z();
        return;
    }

    void KSTrajMagneticDerivative::SetDirectionSign(const int& aSign)
    {
        fDirectionSign = ( aSign < 0 ? -1 : 1 );  // make sure it's either set to -1 or +1
    }

}

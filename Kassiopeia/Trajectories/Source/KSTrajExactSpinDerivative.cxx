#include "KSTrajExactSpinDerivative.h"

namespace Kassiopeia
{

    KSTrajExactSpinDerivative::KSTrajExactSpinDerivative()
    {
    }
    KSTrajExactSpinDerivative::~KSTrajExactSpinDerivative()
    {
    }

    void KSTrajExactSpinDerivative::AddToTime( const double& aTime )
    {
        fData[ 0 ] += aTime;
        return;
    }
    void KSTrajExactSpinDerivative::AddToSpeed( const double& aSpeed )
    {
        fData[ 1 ] += aSpeed;
        return;
    }
    void KSTrajExactSpinDerivative::AddToVelocity( const KThreeVector& aVelocity )
    {
        fData[ 2 ] += aVelocity.X();
        fData[ 3 ] += aVelocity.Y();
        fData[ 4 ] += aVelocity.Z();
        return;
    }
    void KSTrajExactSpinDerivative::AddToForce( const KThreeVector& aForce )
    {
        fData[ 5 ] += aForce.X();
        fData[ 6 ] += aForce.Y();
        fData[ 7 ] += aForce.Z();
        return;
    }
    void KSTrajExactSpinDerivative::AddToOmega0( const double& aOmega0 )
    {
        fData[ 8 ] += aOmega0;
        return;
    }
    void KSTrajExactSpinDerivative::AddToOmega( const KThreeVector& aOmega )
    {
        fData[ 9 ] += aOmega.X();
        fData[ 10 ] += aOmega.Y();
        fData[ 11 ] += aOmega.Z();
        return;
    }
}

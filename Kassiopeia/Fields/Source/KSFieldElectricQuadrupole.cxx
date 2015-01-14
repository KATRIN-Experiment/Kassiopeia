#include "KSFieldElectricQuadrupole.h"

#include <cmath>

namespace Kassiopeia
{

    KSFieldElectricQuadrupole::KSFieldElectricQuadrupole() :
        fLocation( 0., 0., 0. ),
        fStrength( 0. ),
        fLength( 0. ),
        fRadius( 0. ),
        fCharacteristic( 0. )
    {
    }
    KSFieldElectricQuadrupole::KSFieldElectricQuadrupole( const KSFieldElectricQuadrupole& aCopy ) :
        fLocation( aCopy.fLocation ),
        fStrength( aCopy.fStrength ),
        fLength( aCopy.fLength ),
        fRadius( aCopy.fRadius ),
        fCharacteristic( aCopy.fCharacteristic )
    {
    }
    KSFieldElectricQuadrupole* KSFieldElectricQuadrupole::Clone() const
    {
        return new KSFieldElectricQuadrupole( *this );
    }
    KSFieldElectricQuadrupole::~KSFieldElectricQuadrupole()
    {
    }

    void KSFieldElectricQuadrupole::CalculatePotential( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, double& aPotential )
    {
        // thread-safe
        KThreeVector FieldPoint = aSamplePoint - fLocation;
        aPotential = (fStrength / (2. * fCharacteristic * fCharacteristic)) * (FieldPoint[2] * FieldPoint[2] - (1. / 2.) * FieldPoint[0] * FieldPoint[0] - (1. / 2.) * FieldPoint[1] * FieldPoint[1]);
        return;
    }
    void KSFieldElectricQuadrupole::CalculateField( const KThreeVector& aSamplePoint, const double& /*aSampleTime*/, KThreeVector& aField )
    {
        // thread-safe
        KThreeVector FieldPoint = aSamplePoint - fLocation;
        KThreeVector AxialPart = FieldPoint[2] * KThreeVector( 0., 0., 1. );
        KThreeVector RadialPart = FieldPoint - AxialPart;
        aField = (fStrength / (2. * fCharacteristic * fCharacteristic)) * RadialPart - (fStrength / (fCharacteristic * fCharacteristic)) * AxialPart;
        return;
    }

    void KSFieldElectricQuadrupole::SetLocation( const KThreeVector& aLocation )
    {
        fLocation = aLocation;
        return;
    }
    void KSFieldElectricQuadrupole::SetStrength( const double& aStrength )
    {
        fStrength = aStrength;
        return;
    }
    void KSFieldElectricQuadrupole::SetLength( const double& aLength )
    {
        fLength = aLength;
        fCharacteristic = sqrt( (1. / 2.) * (fLength * fLength + (1. / 2.) * fRadius * fRadius) );
        return;
    }
    void KSFieldElectricQuadrupole::SetRadius( const double& aRadius )
    {
        fRadius = aRadius;
        fCharacteristic = sqrt( (1. / 2.) * (fLength * fLength + (1. / 2.) * fRadius * fRadius) );
        return;
    }

}

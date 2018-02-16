/*
 * KRampedElectric2Field.cc
 *
 *  Created on: 16 Jun 2016
 *      Author: wolfgang
 */

#include "KRampedElectric2Field.hh"

#include <assert.h>
#include "KEMCout.hh"
#include "KEMSimpleException.hh"

namespace KEMField {

KRampedElectric2Field::KRampedElectric2Field() :
                       fRootElectricField1( NULL ),
                       fRootElectricField2( NULL ),
                       fRampingType( rtExponential ),
                       fNumCycles( 1 ),
                       fRampUpDelay( 0. ),
                       fRampDownDelay( 0. ),
                       fRampUpTime( 0. ),
                       fRampDownTime( 0. ),
                       fTimeConstant( 0. ),
                       fTimeScalingFactor( 1. )
{
}

KRampedElectric2Field::~KRampedElectric2Field()
{
}

double KRampedElectric2Field::PotentialCore( const KPosition &aSamplePoint, const double &aSampleTime) const
{
    double potField1 = fRootElectricField1->Potential( aSamplePoint, aSampleTime );
    double potField2 = fRootElectricField2->Potential( aSamplePoint, aSampleTime );

    double Modulation = GetModulationFactor( aSampleTime );
    return (1 - Modulation) * potField1 + Modulation * potField2;
    // fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns U=" << aTarget << " at t=" << aSampleTime << eom );
}

KEMThreeVector KRampedElectric2Field::ElectricFieldCore( const KPosition &aSamplePoint, const double &aSampleTime) const
{
    KEMThreeVector field1 =
            fRootElectricField1->ElectricField( aSamplePoint, aSampleTime );
    KEMThreeVector field2 =
            fRootElectricField2->ElectricField( aSamplePoint, aSampleTime );

    double Modulation = GetModulationFactor( aSampleTime );
    return (1 - Modulation) * field1 + Modulation * field2;
    // fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns E=" << aTarget << " at t=" << aSampleTime << eom );
}

double KRampedElectric2Field::GetModulationFactor( const double &aTime ) const
{
    double tLength = fRampUpDelay + fRampUpTime + fRampDownDelay + fRampDownTime;

    double tTime = aTime;
    int tCycle = (int)floor( tTime / tLength );
    if ( tCycle < fNumCycles )
        tTime -= tCycle * tLength;
    tTime *= fTimeScalingFactor;

    double tUp   =         fRampUpDelay   * fTimeScalingFactor;
    double tHigh = tUp   + fRampUpTime    * fTimeScalingFactor;
    double tDown = tHigh + fRampDownDelay * fTimeScalingFactor;
    double tLow  = tDown + fRampDownTime  * fTimeScalingFactor;
    double tOmega = 2.*M_PI/tLength;

    double Field = 0.;
    switch( fRampingType )
    {
    case rtLinear :
        if (tTime >= tLow)
            Field = 0.;
        else if (tTime >= tDown)
            Field = (1. - ((tTime - tDown) / fRampDownTime));
        else if (tTime >= tHigh)
            Field = 1.;
        else if (tTime >= tUp)
            Field = ((tTime - tUp) / fRampUpTime);
        break;

    case rtExponential :
        if (tTime >= tLow)
            Field = 0.;
        else if (tTime >= tDown)
        {
            Field = exp( -(tTime - tDown) / fTimeConstant );
        }
        else if (tTime >= tHigh)
            Field = 1.;
        else if (tTime >= tUp)
            Field = 1. - exp( -(tTime - tUp) / fTimeConstant );
        break;

    case rtSinus :
        if (tTime >= tUp)
            Field = 0.5 + 0.5 * sin(tTime * tOmega);
        break;

    case rtSquare :
        if (tTime >= tDown)
            Field = 0.;
        else if (tTime >= tUp)
            Field = 1.;
        break;

    default :
        throw KEMSimpleException ("Ramped Electric Field: Specified ramping type is not implemented");
        break;
    }

    //fieldmsg_debug( "Ramped electric field <" << GetName() << "> uses modulation factor " << Field << " at t=" << tTime << eom );
    return Field;
}

void KRampedElectric2Field::InitializeCore()
{
    // fieldmsg_debug( "Initializing first root electric field <" << fRootElectricFieldName1 << "> from ramped electric field <" << GetName() << ">" << eom );
    // fieldmsg_debug( "Initializing second root electric field <" << fRootElectricFieldName2 << "> from ramped electric field <" << GetName() << ">" << eom );

    assert( fNumCycles > 0 );
    assert( fTimeScalingFactor > 0. );
    assert( fRampUpTime >= 0. );
    //fieldmsg_assert( fRampUpDelay, >= 0. ); ramping can start before t=0
    assert( fRampDownTime >= 0. );
    //fieldmsg_assert( fRampDownDelay, >= 0. ); ramping back can also start before t=0
    if ( fRampingType == rtExponential )
    {
        assert( fTimeConstant > 0. );
    }

    fRootElectricField1->Initialize();
    fRootElectricField2->Initialize();
}

} /* namespace KEMField */

#include "KSFieldElectricRamped.h"

#include "KToolbox.h"
#include "KSFieldElectromagnet.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{
    KSFieldElectricRamped::KSFieldElectricRamped() :
        fRootElectricField( NULL ),
        fRootElectricFieldName( "" ),
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
    KSFieldElectricRamped::KSFieldElectricRamped( const KSFieldElectricRamped& aCopy ) :
        Kassiopeia::KSComponent(),
        fRootElectricField( aCopy.fRootElectricField ),
        fRootElectricFieldName( aCopy.fRootElectricFieldName ),
        fRampingType( aCopy.fRampingType ),
        fNumCycles( aCopy.fNumCycles ),
        fRampUpDelay( aCopy.fRampUpDelay ),
        fRampDownDelay( aCopy.fRampDownDelay ),
        fRampUpTime( aCopy.fRampUpTime ),
        fRampDownTime( aCopy.fRampDownTime ),
        fTimeConstant( aCopy.fTimeConstant ),
        fTimeScalingFactor( aCopy.fTimeScalingFactor )
    {
    }
    KSFieldElectricRamped* KSFieldElectricRamped::Clone() const
    {
        return new KSFieldElectricRamped( *this );
    }
    KSFieldElectricRamped::~KSFieldElectricRamped()
    {
    }

    void KSFieldElectricRamped::CalculatePotential( const KThreeVector &aSamplePoint, const double &aSampleTime, double &aTarget )
    {
        fRootElectricField->CalculatePotential( aSamplePoint, aSampleTime, aTarget );

        double Modulation = GetModulationFactor( aSampleTime );
        aTarget = aTarget * Modulation;
        fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns U=" << aTarget << " at t=" << aSampleTime << eom );
    }

    void KSFieldElectricRamped::CalculateField( const KThreeVector &aSamplePoint, const double &aSampleTime, KThreeVector &aTarget )
    {
        fRootElectricField->CalculateField( aSamplePoint, aSampleTime, aTarget );

        double Modulation = GetModulationFactor( aSampleTime );
        aTarget = aTarget * Modulation;
        fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns E=" << aTarget << " at t=" << aSampleTime << eom );
    }

    double KSFieldElectricRamped::GetModulationFactor( const double &aTime )
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
				if (tTime >=  tUp)
					Field = sin(tTime*tOmega);
				break;				
					
            default :
                fieldmsg( eError ) << "Specified ramping type is not implemented" << eom;
                break;
        }

        fieldmsg_debug( "Ramped electric field <" << GetName() << "> uses modulation factor " << Field << " at t=" << tTime << eom );
        return Field;
    }

    void KSFieldElectricRamped::InitializeComponent()
    {
        fieldmsg_debug( "Initializing root electric field <" << fRootElectricFieldName << "> from ramped electric field <" << GetName() << ">" << eom );

        fRootElectricField = katrin::KToolbox::GetInstance().Get< KSElectricField >( fRootElectricFieldName );
        if (! fRootElectricField)
        {
            fieldmsg( eError ) << "Ramped electric field <" << GetName() << "> can't find root electric field <" << fRootElectricFieldName << ">!" << eom;
        }

        fieldmsg_assert( fNumCycles, > 0 )
        fieldmsg_assert( fTimeScalingFactor, > 0. )
        fieldmsg_assert( fRampUpTime, >= 0. )
        fieldmsg_assert( fRampUpDelay, >= 0. )
        fieldmsg_assert( fRampDownTime, >= 0. )
        fieldmsg_assert( fRampDownDelay, >= 0. )
        if ( fRampingType == rtExponential )
        {
            fieldmsg_assert( fTimeConstant, > 0. )
        }

        fRootElectricField->Initialize();
    }

    void KSFieldElectricRamped::DeinitializeComponent()
    {
        fRootElectricField->Deinitialize();
    }

}

#include "KSFieldElectricRamped2Fields.h"

#include "KToolbox.h"
#include "KSFieldElectromagnet.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{
    KSFieldElectricRamped2Fields::KSFieldElectricRamped2Fields() :
        fRootElectricField1( NULL ),
        fRootElectricField2( NULL ),
        fRootElectricFieldName1( "" ),
        fRootElectricFieldName2( "" ),
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
    KSFieldElectricRamped2Fields::KSFieldElectricRamped2Fields( const KSFieldElectricRamped2Fields& aCopy ) :
        Kassiopeia::KSComponent(),
        fRootElectricField1( aCopy.fRootElectricField1 ),
        fRootElectricField2( aCopy.fRootElectricField2 ),
        fRootElectricFieldName1( aCopy.fRootElectricFieldName1 ),
        fRootElectricFieldName2( aCopy.fRootElectricFieldName2 ),
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
    KSFieldElectricRamped2Fields* KSFieldElectricRamped2Fields::Clone() const
    {
        return new KSFieldElectricRamped2Fields( *this );
    }
    KSFieldElectricRamped2Fields::~KSFieldElectricRamped2Fields()
    {
    }

    void KSFieldElectricRamped2Fields::CalculatePotential( const KThreeVector &aSamplePoint, const double &aSampleTime, double &aTarget )
    {
        fRootElectricField1->CalculatePotential( aSamplePoint, aSampleTime, aTarget );
        double tempTarget = aTarget;
        fRootElectricField2->CalculatePotential( aSamplePoint, aSampleTime, aTarget );
                
        double Modulation = GetModulationFactor( aSampleTime );
        aTarget = (1 - Modulation) * tempTarget + Modulation * aTarget;
        fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns U=" << aTarget << " at t=" << aSampleTime << eom );
    }

    void KSFieldElectricRamped2Fields::CalculateField( const KThreeVector &aSamplePoint, const double &aSampleTime, KThreeVector &aTarget )
    {
        fRootElectricField1->CalculateField( aSamplePoint, aSampleTime, aTarget );
        KThreeVector tempTarget = aTarget;
        fRootElectricField2->CalculateField( aSamplePoint, aSampleTime, aTarget );
        
        double Modulation = GetModulationFactor( aSampleTime );
        aTarget = (1 - Modulation) * tempTarget + Modulation * aTarget;
        fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns E=" << aTarget << " at t=" << aSampleTime << eom );
    }

    double KSFieldElectricRamped2Fields::GetModulationFactor( const double &aTime )
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
                fieldmsg( eError ) << "Specified ramping type is not implemented" << eom;
                break;
        }

        fieldmsg_debug( "Ramped electric field <" << GetName() << "> uses modulation factor " << Field << " at t=" << tTime << eom );
        return Field;
    }

    void KSFieldElectricRamped2Fields::InitializeComponent()
    {
        fieldmsg_debug( "Initializing first root electric field <" << fRootElectricFieldName1 << "> from ramped electric field <" << GetName() << ">" << eom );
        fieldmsg_debug( "Initializing second root electric field <" << fRootElectricFieldName2 << "> from ramped electric field <" << GetName() << ">" << eom );
        fRootElectricField1 = katrin::KToolbox::GetInstance().Get< KSElectricField >( fRootElectricFieldName1 );
        fRootElectricField2 = katrin::KToolbox::GetInstance().Get< KSElectricField >( fRootElectricFieldName2 );
        if (! fRootElectricField1)
        {
            fieldmsg( eError ) << "Ramped electric field <" << GetName() << "> can't find root electric field <" << fRootElectricFieldName1 << ">!" << eom;
        }
        if (! fRootElectricField2)
        {
            fieldmsg( eError ) << "Ramped electric field <" << GetName() << "> can't find root electric field <" << fRootElectricFieldName2 << ">!" << eom;
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

        fRootElectricField1->Initialize();
        fRootElectricField2->Initialize();
    }

    void KSFieldElectricRamped2Fields::DeinitializeComponent()
    {
        fRootElectricField1->Deinitialize();
        fRootElectricField1->Deinitialize();
    }

}

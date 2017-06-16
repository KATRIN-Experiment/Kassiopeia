#include "KSFieldMagneticRamped.h"

#include "KToolbox.h"
#include "KSFieldElectromagnet.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

namespace Kassiopeia
{
    KSFieldMagneticRamped::KSFieldMagneticRamped() :
        fRootMagneticField( NULL ),
        fRootMagneticFieldName( "" ),
        fRampingType( rtExponential ),
        fNumCycles( 1 ),
        fRampUpDelay( 0. ),
        fRampDownDelay( 0. ),
        fRampUpTime( 0. ),
        fRampDownTime( 0. ),
        fTimeConstant( 1. ),
        fTimeConstant2( 1. ),
        fTimeScalingFactor( 1. )
    {
    }
    KSFieldMagneticRamped::KSFieldMagneticRamped( const KSFieldMagneticRamped& aCopy ) :
        Kassiopeia::KSComponent(),
        fRootMagneticField( aCopy.fRootMagneticField ),
        fRootMagneticFieldName( aCopy.fRootMagneticFieldName ),
        fRampingType( aCopy.fRampingType ),
        fNumCycles( aCopy.fNumCycles ),
        fRampUpDelay( aCopy.fRampUpDelay ),
        fRampDownDelay( aCopy.fRampDownDelay ),
        fRampUpTime( aCopy.fRampUpTime ),
        fRampDownTime( aCopy.fRampDownTime ),
        fTimeConstant( aCopy.fTimeConstant ),
        fTimeConstant2( aCopy.fTimeConstant2 ),
        fTimeScalingFactor( aCopy.fTimeScalingFactor )
    {
    }
    KSFieldMagneticRamped* KSFieldMagneticRamped::Clone() const
    {
        return new KSFieldMagneticRamped( *this );
    }
    KSFieldMagneticRamped::~KSFieldMagneticRamped()
    {
    }

    void KSFieldMagneticRamped::CalculateField( const KThreeVector &aSamplePoint, const double &aSampleTime, KThreeVector &aTarget )
    {
        fRootMagneticField->CalculateField( aSamplePoint, aSampleTime, aTarget );

        double Modulation = GetModulationFactor( aSampleTime );
        aTarget = aTarget * Modulation;
        fieldmsg_debug( "Ramped magnetic field <" << GetName() << "> returns B=" << aTarget << " at t=" << aSampleTime << eom );
    }

    void KSFieldMagneticRamped::CalculateGradient( const KThreeVector &aSamplePoint, const double &aSampleTime, KThreeMatrix &aTarget )
    {
        fRootMagneticField->CalculateGradient( aSamplePoint, aSampleTime, aTarget );

        double Modulation = GetModulationFactor( aSampleTime );
        aTarget = aTarget * Modulation;
        fieldmsg_debug( "Ramped magnetic field <" << GetName() << "> returns gradB=" << aTarget << " at t=" << aSampleTime << eom );
    }

    double KSFieldMagneticRamped::GetModulationFactor( const double &aTime )
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
                    Field = exp( -(tTime - tDown) / fTimeConstant );
                else if (tTime >= tHigh)
                    Field = 1.;
                else if (tTime >= tUp)
                    Field = 1. - exp( -(tTime - tUp) / fTimeConstant );
                break;

            case rtInversion :
                if (tTime >= tLow)
                    Field = 1.;
                else if (tTime >= tDown)  // back to normal
                    Field = 1. - 2. * exp( -(tTime - tDown) / fTimeConstant );
                else if (tTime >= tHigh)
                    Field = -1.;
                else if (tTime >= tUp)  // to inverted
                    Field = -1. + 2. * exp( -(tTime - tUp) / fTimeConstant );
                break;

            case rtInversion2 :
                if (tTime >= tLow)
                    Field = 1.;
                else if (tTime >= tDown)  // back to normal
                    Field = 1. - ( exp( -(tTime - tDown) / fTimeConstant ) + exp( -(tTime - tDown) / fTimeConstant2 ) );
                else if (tTime >= tHigh)
                    Field = -1.;
                else if (tTime >= tUp)  // to inverted
                    Field = -1. + ( exp( -(tTime - tUp) / fTimeConstant ) + exp( -(tTime - tUp) / fTimeConstant2 ) );
                break;

            case rtFlipBox :
                if (tTime < tDown)  // normal
                    Field = 1.;
                else if (tTime >= tDown)  // ramp to normal
                {
                    double tMid = 0.5 * (tLow + tDown);
                    if (tTime < tMid)
                        Field = -1. * exp( -(tTime - tDown) / fTimeConstant );
                    else
                        Field = 1. - exp( -(tTime - tMid) / fTimeConstant );
                }
                else if (tTime >= tHigh)  // inverted
                    Field = -1.;
                else if (tTime >= tUp)  // ramp to inverted
                {
                    double tMid = 0.5 * (tHigh + tUp);
                    if (tTime < tMid)
                        Field = exp( -(tTime - tUp) / fTimeConstant );
                    else
                        Field = -1. + exp( -(tTime - tMid) / fTimeConstant );
                }
                else if (tTime >= tLow)
                    Field = 1.;
                break;

            default :
                fieldmsg( eError ) << "Specified ramping type is not implemented" << eom;
                break;
        }

        // magnetic field must not be zero
        if ( Field == 0. )
            Field = 1e-12;  // use a small number, but not TOO small

        fieldmsg_debug( "Ramped magnetic field <" << GetName() << "> uses modulation factor " << Field << " at t=" << tTime << eom );
        return Field;
    }

    double KSFieldMagneticRamped::GetDerivModulationFactor( const double &aTime )
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

        double DerivField = 0.;
        switch( fRampingType )
        {
            case rtLinear :
                if (tTime >= tLow)
                    DerivField = 0.;
                else if (tTime >= tDown)
                    DerivField = (-1. / fRampDownTime);
                else if (tTime >= tHigh)
                    DerivField = 0.;
                else if (tTime >= tUp)
                    DerivField = (1. / fRampUpTime);
                break;

            case rtExponential :
                if (tTime >= tLow)
                    DerivField = 0.;
                else if (tTime >= tDown)
                    DerivField = 1. / fTimeConstant * exp( -(tTime - tDown) / fTimeConstant );
                else if (tTime >= tHigh)
                    DerivField = 0.;
                else if (tTime >= tUp)
                    DerivField = -1. / fTimeConstant * exp( -(tTime - tUp) / fTimeConstant );
                break;

            case rtInversion :
                if (tTime >= tLow)
                    DerivField = 0.;
                else if (tTime >= tDown)
                    DerivField = 2. / fTimeConstant * exp( -(tTime - tDown) / fTimeConstant );
                else if (tTime >= tHigh)
                    DerivField = 0.;
                else if (tTime >= tUp)
                    DerivField = -2. / fTimeConstant * exp( -(tTime - tUp) / fTimeConstant );
                break;

            case rtInversion2 :
                if (tTime >= tLow)
                    DerivField = 0.;
                else if (tTime >= tDown)
                    DerivField = 1. / fTimeConstant * exp( -(tTime - tDown) / fTimeConstant ) + 1. / fTimeConstant2 * exp( -(tTime - tDown) / fTimeConstant2 );
                else if (tTime >= tHigh)
                    DerivField = 0.;
                else if (tTime >= tUp)
                    DerivField = -1. / fTimeConstant * exp( -(tTime - tUp) / fTimeConstant ) - 1. / fTimeConstant2 * exp( -(tTime - tUp) / fTimeConstant2 );
                break;

            case rtFlipBox :
                if (tTime >= tLow)  // normal
                    DerivField = 0.;
                else if (tTime >= tDown)  // ramp to normal
                {
                    double tMid = 0.5 * (tLow + tDown);
                    if (tTime < tMid)
                        DerivField = -1. / fTimeConstant * exp( -(tTime - tDown) / fTimeConstant );
                    else
                        DerivField = -1. / fTimeConstant * exp( -(tTime - tMid) / fTimeConstant );
                }
                else if (tTime >= tHigh)  // inverted
                    DerivField = 0.;
                else if (tTime >= tUp)  // ramp to inverted
                {
                    double tMid = 0.5 * (tHigh + tUp);
                    if (tTime < tMid)
                        DerivField = 1. / fTimeConstant * exp( -(tTime - tUp) / fTimeConstant );
                    else
                        DerivField = 1. / fTimeConstant * exp( -(tTime - tMid) / fTimeConstant );
                }
                break;

            default :
                fieldmsg( eError ) << "Specified ramping type is not implemented" << eom;
                //break;
        }

        fieldmsg_debug( "Ramped magnetic field <" << GetName() << "> uses derived modulation factor " << DerivField << " at t=" << tTime << eom );
        return DerivField;
    }

    void KSFieldMagneticRamped::InitializeComponent()
    {
        fieldmsg_debug( "Initializing root magnetic field <" << fRootMagneticFieldName << "> from ramped magnetic field <" << GetName() << ">" << eom );

        fRootMagneticField = katrin::KToolbox::GetInstance().Get< KSMagneticField >( fRootMagneticFieldName );
        if (! fRootMagneticField)
        {
            fieldmsg( eError ) << "Ramped magnetic field <" << GetName() << "> can't find root magnetic field <" << fRootMagneticFieldName << ">!" << eom;
        }

        fieldmsg_assert( fNumCycles, > 0 )
        fieldmsg_assert( fTimeScalingFactor, > 0. )
        fieldmsg_assert( fRampUpTime, >= 0. )
        fieldmsg_assert( fRampUpDelay, >= 0. )
        fieldmsg_assert( fRampDownTime, >= 0. )
        fieldmsg_assert( fRampDownDelay, >= 0. )
        if ( fRampingType == rtExponential || fRampingType == rtInversion || fRampingType == rtInversion2 || fRampingType == rtFlipBox )
        {
            fieldmsg_assert( fTimeConstant, > 0. )
        }
        if ( fRampingType == rtInversion2 )
        {
            fieldmsg_assert( fTimeConstant2, > 0. )
        }

        fRootMagneticField->Initialize();
    }

    void KSFieldMagneticRamped::DeinitializeComponent()
    {
        fRootMagneticField->Deinitialize();
    }

    /////////////////////////////////////////////////////////////////////////

    KSFieldElectricInducedAzimuthal::KSFieldElectricInducedAzimuthal() :
        fRampedMagneticField( NULL ),
        fRampedMagneticFieldName( "" )
    {
    }
    KSFieldElectricInducedAzimuthal::KSFieldElectricInducedAzimuthal( const KSFieldElectricInducedAzimuthal& aCopy ) :
        Kassiopeia::KSComponent(),
        fRampedMagneticField( aCopy.fRampedMagneticField ),
        fRampedMagneticFieldName( aCopy.fRampedMagneticFieldName )
    {
    }
    KSFieldElectricInducedAzimuthal* KSFieldElectricInducedAzimuthal::Clone() const
    {
        return new KSFieldElectricInducedAzimuthal( *this );
    }
    KSFieldElectricInducedAzimuthal::~KSFieldElectricInducedAzimuthal()
    {
    }

    void KSFieldElectricInducedAzimuthal::CalculatePotential( const KThreeVector &aSamplePoint, const double &aSampleTime, double &aTarget )
    {
        KThreeVector tElectricField;
        CalculateField( aSamplePoint, aSampleTime, tElectricField );

        aTarget = -1. * tElectricField.Dot( aSamplePoint );
        fieldmsg_debug( "Induced azimuthal electric field <" << GetName() << "> returns U=" << aTarget << " at t=" << aSampleTime << eom );
    }

    void KSFieldElectricInducedAzimuthal::CalculateField( const KThreeVector &aSamplePoint, const double &aSampleTime, KThreeVector &aTarget )
    {
        double tRadius = aSamplePoint.Perp();
        if (tRadius > 0.)
        {
            KThreeVector tAziDirection = 1. / tRadius * KThreeVector( -aSamplePoint.Y(), aSamplePoint.X(), 0. );

            KThreeVector tMagneticField;
            fRampedMagneticField->CalculateField( aSamplePoint, aSampleTime, tMagneticField );

            double Modulation = fRampedMagneticField->GetDerivModulationFactor( aSampleTime );
            aTarget = tAziDirection * (tMagneticField.Z() * (-tRadius / 2.)) * (Modulation * fRampedMagneticField->GetTimeScalingFactor());
        }
        else
        {
            aTarget.SetComponents(0.,0.,0.);
        }
        fieldmsg_debug( "Induced azimuthal electric field <" << GetName() << "> returns E=" << aTarget << " at t=" << aSampleTime << " with r=" << tRadius << eom );
    }

    void KSFieldElectricInducedAzimuthal::InitializeComponent()
    {
        fieldmsg_debug( "Initializing ramped magnetic field <" << fRampedMagneticFieldName << "> from induced azimuthal electric field <" << GetName() << ">" << eom );

        fRampedMagneticField = katrin::KToolbox::GetInstance().Get< KSFieldMagneticRamped >( fRampedMagneticFieldName );
        if (! fRampedMagneticField)
        {
            fieldmsg( eError ) << "Induced azimuthal electric field <" << GetName() << "> can't find ramped magnetic field <" << fRampedMagneticFieldName << ">!" << eom;
        }

        // fRampedMagneticField should already be initialized
    }

    void KSFieldElectricInducedAzimuthal::DeinitializeComponent()
    {
        // fRampedMagneticField should be deinitialized later
    }

}

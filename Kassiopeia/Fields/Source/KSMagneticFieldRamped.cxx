#include "KSMagneticFieldRamped.h"

namespace Kassiopeia
{
    KSMagneticFieldRamped::KSMagneticFieldRamped() :
            fRootMagneticField( NULL ),
            fRampingType( rtExponential ),
            fRampUpDelay( 0. ),
            fRampDownDelay( 0. ),
            fRampUpTime( 0. ),
            fRampDownTime( 0. ),
            fTimeConstant( 0. ),
            fTimeScalingFactor( 1. ),
            fMaxFieldFactor( 0. )
    {
    }

    KSMagneticFieldRamped::~KSMagneticFieldRamped()
    {
    }

    KSMagneticField* KSMagneticFieldRamped::GetSourceField()
    {
        return fRootMagneticField;
    }

    bool KSMagneticFieldRamped::GetField( KThreeVector &aTarget, const KThreeVector &aSamplePoint, const Double_t &aSampleTime )
    {
        // NOT THREAD-SAFE
        bool tValid = GetSourceField()->GetField( aTarget, aSamplePoint, aSampleTime );
        double Modulation = GetModulationFactor( aSampleTime );

        aTarget = aTarget * Modulation;

        return tValid;
    }

    bool KSMagneticFieldRamped::GetGradient( KThreeMatrix &aTarget, const KThreeVector &aSamplePoint, const Double_t &aSampleTime )
    {
        // NOT THREAD-SAFE
        bool tValid = GetSourceField()->GetGradient( aTarget, aSamplePoint, aSampleTime );
        double Modulation = GetModulationFactor( aSampleTime );

        aTarget = aTarget * Modulation;

        return tValid;
    }

    double KSMagneticFieldRamped::GetModulationFactor( const double &aTime )
    {
        double tUp = fRampUpDelay;
        double tHigh = tUp + fRampUpTime;
        double tDown = tHigh + fRampDownDelay;
        double tLow = tDown + fRampDownTime;

        if( fabs( fTimeScalingFactor ) <= 0. )
        {
            fieldmsg( eError ) << "Ramped magnetic field <" << GetName() << "> can't use a negative time scaling factor!" << eom;
        }

        double tTime = aTime * fTimeScalingFactor;
        double Field = 0.;

        switch( fRampingType )
        {
            case rtLinear :
                if( (fRampUpTime > 0.) && (tTime >= tUp) && (tTime < tHigh) )
                {
                    Field = ((tTime - tUp) / fRampUpTime);
                }
                else if( (fRampUpTime > 0.) && (tTime >= tHigh) && (tTime < tDown) )
                {
                    Field = 1.;
                }
                else if( (fRampDownTime > 0.) && (tTime >= tDown) && (tTime < tLow) )
                {
                    Field = (1. - ((tTime - tDown) / fRampDownTime));
                }
                else
                {
                    Field = 0.;
                }
                break;

            case rtExponential :
                if( (fRampUpTime > 0.) && (tTime >= tUp) && (tTime < tHigh) )
                {
                    Field = (1. - exp( -(tTime - tUp) / fTimeConstant ));
                    // save the maximum achieved field for the ramp-down phase
                    // this implementation is correct since the field is strictly increasing in ramp-up
                    if( fabs( Field ) > 0. )
                    {
                        /// FIXME - this code makes the function not thread-safe
                        fMaxFieldFactor = Field;
                    }
                }
                else if( (fRampUpTime > 0.) && (tTime >= tHigh) && (tTime < tDown) )
                {
                    /// FIXME - this code makes the function not thread-safe
                    Field = 1.;
                    fMaxFieldFactor = 1.;
                }
                else if( (fRampDownTime > 0.) && (tTime >= tDown) && (tTime < tLow) )
                {
                    Field = fMaxFieldFactor * exp( -(tTime - tDown) / fTimeConstant );
                    // do not set fMaxFieldFactor here
                }
                else
                {
                    /// FIXME - this code makes the function not thread-safe
                    Field = 0.;
                    fMaxFieldFactor = 0.;
                }
                break;

            default :
                fieldmsg( eError ) << "Specified ramping type is not implemented" << eom;
                break;
        }

        return Field;
    }

    double KSMagneticFieldRamped::GetDerivModulationFactor( const double &aTime )
    {
        double tUp = fRampUpDelay;
        double tHigh = tUp + fRampUpTime;
        double tDown = tHigh + fRampDownDelay;
        double tLow = tDown + fRampDownTime;

        if( fabs( fTimeScalingFactor ) <= 0. )
        {
            fieldmsg( eError ) << "Ramped magnetic field <" << GetName() << "> can't use a negative time scaling factor!" << eom;
        }

        double tTime = aTime * fTimeScalingFactor;
        double DerivField = 0.;

        switch( fRampingType )
        {
            case rtLinear :
                if( (fRampUpTime > 0.) && (tTime >= tUp) && (tTime < tHigh) )
                {
                    DerivField = (1. / fRampUpTime);
                }
                else if( (fRampUpTime > 0.) && (tTime >= tHigh) && (tTime < tDown) )
                {
                    DerivField = 0.;
                }
                else if( (fRampDownTime > 0.) && (tTime >= tDown) && (tTime < tLow) )
                {
                    DerivField = (-1. / fRampDownTime);
                }
                else
                {
                    DerivField = 0.;
                }
                break;

            case rtExponential :
                if( (fRampUpTime > 0.) && (tTime >= tUp) && (tTime < tHigh) )
                {
                    DerivField = (1. / fTimeConstant * exp( -(tTime - tUp) / fTimeConstant ));
                }
                else if( (fRampUpTime > 0.) && (tTime >= tHigh) && (tTime < tDown) )
                {
                    DerivField = 0.;
                }
                else if( (fRampDownTime > 0.) && (tTime >= tDown) && (tTime < tLow) )
                {
                    DerivField = fMaxFieldFactor * (-1. / fTimeConstant * exp( -(tTime - (tDown)) / fTimeConstant ));
                }
                else
                {
                    DerivField = 0.;
                }
                break;

            default :
                fieldmsg( eError ) << "Specified ramping type is not implemented" << eom;
                break;
        }

        return DerivField;
    }

    /////////////////////////////////////////////////////////////////////////

    KSInducedAzimuthalElectricField::KSInducedAzimuthalElectricField() :
            fRampedMagneticField( NULL )
    {
    }

    KSInducedAzimuthalElectricField::~KSInducedAzimuthalElectricField()
    {
    }

    bool KSInducedAzimuthalElectricField::GetPhi( Double_t &aTarget, const KThreeVector &aSamplePoint, const Double_t &aSampleTime )
    {
        // NOT THREAD-SAFE
        KThreeVector tElectricField;

        bool tValid = GetField( tElectricField, aSamplePoint, aSampleTime );
        aTarget = -1. * tElectricField.Dot( aSamplePoint );

        return tValid;
    }

    bool KSInducedAzimuthalElectricField::GetField( KThreeVector &aTarget, const KThreeVector &aSamplePoint, const Double_t &aSampleTime )
    {
        // NOT THREAD-SAFE
        double tRadius = aSamplePoint.Perp();
        KThreeVector tAziDirection = 1. / tRadius * KThreeVector( -aSamplePoint.Y(), aSamplePoint.X(), 0. );
        KThreeVector tMagneticField;

        bool tValid = fRampedMagneticField->GetField( tMagneticField, aSamplePoint, aSampleTime );
        double Modulation = fRampedMagneticField->GetDerivModulationFactor( aSampleTime );

        aTarget = tAziDirection * (tMagneticField.Z() * (-tRadius / 2.)) * (Modulation * fRampedMagneticField->GetTimeScalingFactor());

        return tValid;
    }

    bool KSInducedAzimuthalElectricField::GetGradient( KThreeMatrix &/*aTarget*/, const KThreeVector &/*aSamplePoint*/, const Double_t &/*aSampleTime*/)
    {
        return false;
    }

}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

#include "KSToolbox.h"

namespace katrin
{

    template< >
    KSRampedMagneticFieldBuilder::~KComplexElement()
    {
    }

    static int KSRampedMagneticFieldStructure =
        KSRampedMagneticFieldBuilder::Attribute< string >( "name" ) +
        KSRampedMagneticFieldBuilder::Attribute< string >( "root_field" ) +
        KSRampedMagneticFieldBuilder::Attribute< string >( "ramping_type" ) +
        KSRampedMagneticFieldBuilder::Attribute< double >( "ramp_up_delay" ) +
        KSRampedMagneticFieldBuilder::Attribute< double >( "ramp_down_delay" ) +
        KSRampedMagneticFieldBuilder::Attribute< double >( "ramp_up_time" ) +
        KSRampedMagneticFieldBuilder::Attribute< double >( "ramp_down_time" ) +
        KSRampedMagneticFieldBuilder::Attribute< double >( "time_constant" ) +
        KSRampedMagneticFieldBuilder::Attribute< double >( "time_scaling" );

    static int sKSRampedMagneticField =
        KSToolboxBuilder::ComplexElement< KSMagneticFieldRamped >( "ramped_magnetic_field" );

}

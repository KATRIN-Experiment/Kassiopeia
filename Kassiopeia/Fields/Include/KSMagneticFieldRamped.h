#ifndef Kassiopeia_KSMagneticFieldRamped_h_
#define Kassiopeia_KSMagneticFieldRamped_h_

#include "KSMagneticField.h"
#include "KSElectricField.h"

#include "KSToolbox.h"
#include "KSFieldMessage.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSMagneticFieldRamped :
    public KSComponentTemplate< KSMagneticFieldRamped, KSMagneticField >
    {
        public:
            typedef enum {
                rtLinear,
                rtExponential
            } eRampingType;

        public:
            KSMagneticFieldRamped();
            virtual ~KSMagneticFieldRamped();

        public:
            virtual bool GetField( KThreeVector& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );
            virtual bool GetGradient( KThreeMatrix& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );

        public:
            KSMagneticField* GetSourceField();

            double GetModulationFactor( const double& aTime );
            double GetDerivModulationFactor( const double& aTime );

        public:
            ;K_SET_GET_PTR( KSMagneticField, RootMagneticField )
            ;K_SET_GET( eRampingType, RampingType )
            ;K_SET_GET( double, RampUpDelay )
            ;K_SET_GET( double, RampDownDelay )
            ;K_SET_GET( double, RampUpTime )
            ;K_SET_GET( double, RampDownTime )
            ;K_SET_GET( double, TimeConstant )
            ;K_SET_GET( double, TimeScalingFactor )

        private:
            double fMaxFieldFactor;
    };

    /////////////////////////////////////////////////////////////////////////

    class KSInducedAzimuthalElectricField :
        public KSElectricField
    {
        public:
            KSInducedAzimuthalElectricField();
            virtual ~KSInducedAzimuthalElectricField();

        public:
            virtual bool GetPhi( Double_t& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );
            virtual bool GetField( KThreeVector& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );
            virtual bool GetGradient( KThreeMatrix& aTarget, const KThreeVector& aSamplePoint, const Double_t& aSampleTime );

        public:
            ;K_SET_PTR( KSMagneticFieldRamped, RampedMagneticField )
    };

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


#include "KComplexElement.hh"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSMagneticFieldRamped > KSRampedMagneticFieldBuilder;

    template< >
    inline bool KSRampedMagneticFieldBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSMagneticField::SetName );
            return true;
        }
        if( aContainer->GetName() == "root_field" )
        {
            KSMagneticField* tRootField = KSToolbox::GetInstance()->GetObjectAs<KSMagneticField>( aContainer->AsReference< string >() );
            fObject->SetRootMagneticField( tRootField );
            return true;
        }
        if( aContainer->GetName() == "ramping_type" )
        {
            const string& tName = aContainer->AsReference< string >();
            if( (tName == "linear") || (tName == "lin") )
            {
                fObject->SetRampingType( KSMagneticFieldRamped::rtLinear );
                return true;
            }
            if( (tName == "exponential") || (tName == "exp") )
            {
                fObject->SetRampingType( KSMagneticFieldRamped::rtExponential );
                return true;
            }
            fieldmsg( eWarning ) << "specified ramping type <" << tName << "> can only be one of <linear>, <lin>, <exponential>, <exp>" << eom;
            return false;
        }
        if( aContainer->GetName() == "ramp_up_delay" )
        {
            aContainer->CopyTo( fObject, &KSMagneticFieldRamped::SetRampUpDelay );
            return true;
        }
        if( aContainer->GetName() == "ramp_down_delay" )
        {
            aContainer->CopyTo( fObject, &KSMagneticFieldRamped::SetRampDownDelay );
            return true;
        }
        if( aContainer->GetName() == "ramp_up_time" )
        {
            aContainer->CopyTo( fObject, &KSMagneticFieldRamped::SetRampUpTime );
            return true;
        }
        if( aContainer->GetName() == "ramp_down_time" )
        {
            aContainer->CopyTo( fObject, &KSMagneticFieldRamped::SetRampDownTime );
            return true;
        }
        if( aContainer->GetName() == "time_constant" )
        {
            aContainer->CopyTo( fObject, &KSMagneticFieldRamped::SetTimeConstant );
            return true;
        }
        if( aContainer->GetName() == "time_scaling" )
        {
            aContainer->CopyTo( fObject, &KSMagneticFieldRamped::SetTimeScalingFactor );
            return true;
        }
        return false;
    }


/*
    template< >
    inline bool KSRampedMagneticFieldBuilder::End()
    {

        KSFieldToolbox* tFieldToolbox = KSFieldToolbox::GetInstance();

        fieldmsg( eNormal ) << "Automatically adding induced electric field to ramped magnetic field <" << fObject->GetName() << ">" << eom;
        KSInducedAzimuthalElectricField* tInducedElectricField = new KSInducedAzimuthalElectricField();
        tInducedElectricField->SetName( fObject->GetName() + string("_induced_efield") );
        tInducedElectricField->SetRampedMagneticField( fObject );
        tFieldToolbox->AddObject< KSElectricField >( tInducedElectricField );

        return true;
    }
*/


}

#endif

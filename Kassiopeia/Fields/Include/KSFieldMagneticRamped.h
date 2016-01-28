#ifndef Kassiopeia_KSMagneticFieldRamped_h_
#define Kassiopeia_KSMagneticFieldRamped_h_

#include "KSMagneticField.h"
#include "KSElectricField.h"

#include "KSFieldsMessage.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSFieldMagneticRamped :
    public KSComponentTemplate< KSFieldMagneticRamped, KSMagneticField >
    {
        public:
            typedef enum {
                rtLinear,           // simple linear ramping
                rtExponential,      // exponential ramping with given time constant
                rtInversion,        // ramping to inverted magnetic field using single exponential ramping
                rtInversion2,       // ramping to inverted magnetic field using exponential ramping with two time constants
                rtFlipBox,          // ramping to inverted magnetic field using double exponential ramping (ramp to zero in between)
            } eRampingType;

        public:
            KSFieldMagneticRamped();
            KSFieldMagneticRamped( const KSFieldMagneticRamped& aCopy );
            KSFieldMagneticRamped* Clone() const;
            virtual ~KSFieldMagneticRamped();

        public:
            virtual void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aTarget );
            virtual void CalculateGradient( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeMatrix& aTarget );

        public:
            double GetModulationFactor( const double& aTime );
            double GetDerivModulationFactor( const double& aTime );

        protected:
            virtual void InitializeComponent();
            virtual void DeinitializeComponent();

        public:
            ;K_SET_GET_PTR( KSMagneticField, RootMagneticField )
            ;K_SET_GET( string, RootMagneticFieldName )
            ;K_SET_GET( eRampingType, RampingType )
            ;K_SET_GET( int, NumCycles )
            ;K_SET_GET( double, RampUpDelay )
            ;K_SET_GET( double, RampDownDelay )
            ;K_SET_GET( double, RampUpTime )
            ;K_SET_GET( double, RampDownTime )
            ;K_SET_GET( double, TimeConstant )
            ;K_SET_GET( double, TimeConstant2 )
            ;K_SET_GET( double, TimeScalingFactor )

    };

    /////////////////////////////////////////////////////////////////////////

    class KSFieldElectricInducedAzimuthal :
        public KSComponentTemplate< KSFieldElectricInducedAzimuthal, KSElectricField >
    {
        public:
            KSFieldElectricInducedAzimuthal();
            KSFieldElectricInducedAzimuthal( const KSFieldElectricInducedAzimuthal& aCopy );
            KSFieldElectricInducedAzimuthal* Clone() const;
            virtual ~KSFieldElectricInducedAzimuthal();

        public:
            virtual void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aTarget );
            virtual void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aTarget );

        protected:
            virtual void InitializeComponent();
            virtual void DeinitializeComponent();

        public:
            ;K_SET_GET_PTR( KSFieldMagneticRamped, RampedMagneticField )
            ;K_SET_GET( string, RampedMagneticFieldName )
    };

}

#endif

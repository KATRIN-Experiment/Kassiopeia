#ifndef Kassiopeia_KSElectricFieldRamped_h_
#define Kassiopeia_KSElectricFieldRamped_h_

#include "KSElectricField.h"

#include "KSFieldsMessage.h"

#include "KField.h"

namespace Kassiopeia
{

    class KSFieldElectricRamped :
    public KSComponentTemplate< KSFieldElectricRamped, KSElectricField >
    {
        public:
            typedef enum {
                rtLinear,           // simple linear ramping
                rtExponential,      // exponential ramping with given time constant
            } eRampingType;

        public:
            KSFieldElectricRamped();
            KSFieldElectricRamped( const KSFieldElectricRamped& aCopy );
            KSFieldElectricRamped* Clone() const;
            virtual ~KSFieldElectricRamped();

        public:
            virtual void CalculatePotential( const KThreeVector& aSamplePoint, const double& aSampleTime, double& aTarget );
            virtual void CalculateField( const KThreeVector& aSamplePoint, const double& aSampleTime, KThreeVector& aTarget );

        public:
            double GetModulationFactor( const double& aTime );

        protected:
            virtual void InitializeComponent();
            virtual void DeinitializeComponent();

        public:
            ;K_SET_GET_PTR( KSElectricField, RootElectricField )
            ;K_SET_GET( string, RootElectricFieldName )
            ;K_SET_GET( eRampingType, RampingType )
            ;K_SET_GET( int, NumCycles )
            ;K_SET_GET( double, RampUpDelay )
            ;K_SET_GET( double, RampDownDelay )
            ;K_SET_GET( double, RampUpTime )
            ;K_SET_GET( double, RampDownTime )
            ;K_SET_GET( double, TimeConstant )
            ;K_SET_GET( double, TimeScalingFactor )

    };

}

#endif

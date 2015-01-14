#ifndef KSDIMENSIONS_H_
#define KSDIMENSIONS_H_

#include "KSAssert.h"

#include "KTypeList.h"
#include "KTypeInteger.h"
#include "KTypeLogic.h"

#include <string>
using std::string;

#include <sstream>
using std::stringstream;

namespace katrin
{

    //********************
    //dimension definition
    //********************

    template< class XDimensionList >
    class KDimension;

    template< int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue, int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue >
    class KDimension< KTYPELIST7( KTypeInteger< XMassPowerValue >, KTypeInteger< XLengthPowerValue >, KTypeInteger< XTimePowerValue >, KTypeInteger< XChargePowerValue >, KTypeInteger< XTemperaturePowerValue >, KTypeInteger< XIntensityPowerValue >, KTypeInteger< XAnglePowerValue > ) >
    {
        public:
            template< class XRight >
            class Multiply;

            template< int XRightMassPowerValue, int XRightLengthPowerValue, int XRightTimePowerValue, int XRightChargePowerValue, int XRightTemperaturePowerValue, int XRightIntensityPowerValue, int XRightAnglePowerValue >
            class Multiply< KDimension< KTYPELIST7( KTypeInteger< XRightMassPowerValue >, KTypeInteger< XRightLengthPowerValue >, KTypeInteger< XRightTimePowerValue >, KTypeInteger< XRightChargePowerValue >, KTypeInteger< XRightTemperaturePowerValue >, KTypeInteger< XRightIntensityPowerValue >, KTypeInteger< XRightAnglePowerValue > ) > >
            {
                public:
                    typedef KDimension< KTYPELIST7( KTypeInteger< XMassPowerValue + XRightMassPowerValue >, KTypeInteger< XLengthPowerValue + XRightLengthPowerValue >, KTypeInteger< XTimePowerValue + XRightTimePowerValue >, KTypeInteger< XChargePowerValue + XRightChargePowerValue >, KTypeInteger< XTemperaturePowerValue + XRightTemperaturePowerValue >, KTypeInteger< XIntensityPowerValue + XRightIntensityPowerValue >, KTypeInteger< XAnglePowerValue + XRightAnglePowerValue > ) > Type;
            };

            template< class XRight >
            class Divide;

            template< int XRightMassPowerValue, int XRightLengthPowerValue, int XRightTimePowerValue, int XRightChargePowerValue, int XRightTemperaturePowerValue, int XRightIntensityPowerValue, int XRightAnglePowerValue >
            class Divide< KDimension< KTYPELIST7( KTypeInteger< XRightMassPowerValue >, KTypeInteger< XRightLengthPowerValue >, KTypeInteger< XRightTimePowerValue >, KTypeInteger< XRightChargePowerValue >, KTypeInteger< XRightTemperaturePowerValue >, KTypeInteger< XRightIntensityPowerValue >, KTypeInteger< XRightAnglePowerValue > ) > >
            {
                public:
                    typedef KDimension< KTYPELIST7( KTypeInteger< XMassPowerValue - XRightMassPowerValue >, KTypeInteger< XLengthPowerValue - XRightLengthPowerValue >, KTypeInteger< XTimePowerValue - XRightTimePowerValue >, KTypeInteger< XChargePowerValue - XRightChargePowerValue >, KTypeInteger< XTemperaturePowerValue - XRightTemperaturePowerValue >, KTypeInteger< XIntensityPowerValue - XRightIntensityPowerValue >, KTypeInteger< XAnglePowerValue - XRightAnglePowerValue > ) > Type;
            };

            template< class XRight >
            class Equal;

            template< int XRightMassPowerValue, int XRightLengthPowerValue, int XRightTimePowerValue, int XRightChargePowerValue, int XRightTemperaturePowerValue, int XRightIntensityPowerValue, int XRightAnglePowerValue >
            class Equal< KDimension< KTYPELIST7( KTypeInteger< XRightMassPowerValue >, KTypeInteger< XRightLengthPowerValue >, KTypeInteger< XRightTimePowerValue >, KTypeInteger< XRightChargePowerValue >, KTypeInteger< XRightTemperaturePowerValue >, KTypeInteger< XRightIntensityPowerValue >, KTypeInteger< XRightAnglePowerValue > ) > >
            {
                public:
                     enum
                    {
                        Value = KSTypeEqual< KDimension< KTYPELIST7( KTypeInteger< XMassPowerValue >, KTypeInteger< XLengthPowerValue >, KTypeInteger< XTimePowerValue >, KTypeInteger< XChargePowerValue >, KTypeInteger< XTemperaturePowerValue >, KTypeInteger< XIntensityPowerValue >, KTypeInteger< XAnglePowerValue > ) >, KDimension< KTYPELIST7( KTypeInteger< XRightMassPowerValue >, KTypeInteger< XRightLengthPowerValue >, KTypeInteger< XRightTimePowerValue >, KTypeInteger< XRightChargePowerValue >, KTypeInteger< XRightTemperaturePowerValue >, KTypeInteger< XRightIntensityPowerValue >, KTypeInteger< XRightAnglePowerValue > ) > >::Value
                    };
            };

        public:
            static const string fSymbol;
            static const string ConstructDimensionSymbol()
            {

                stringstream Symbol;
                Symbol.clear();
                Symbol.str( "" );

                Symbol << "[ ";

                if( XMassPowerValue != 0 )
                {
                    Symbol << "M";
                    if( XMassPowerValue != 1 )
                    {
                        Symbol << "^" << XMassPowerValue;
                    }
                    Symbol << " ";
                }

                if( XLengthPowerValue != 0 )
                {
                    Symbol << "L";
                    if( XLengthPowerValue != 1 )
                    {
                        Symbol << "^" << XLengthPowerValue;
                    }
                    Symbol << " ";
                }

                if( XTimePowerValue != 0 )
                {
                    Symbol << "T";
                    if( XTimePowerValue != 1 )
                    {
                        Symbol << "^" << XTimePowerValue;
                    }
                    Symbol << " ";
                }

                if( XChargePowerValue != 0 )
                {
                    Symbol << "Q";
                    if( XChargePowerValue != 1 )
                    {
                        Symbol << "^" << XChargePowerValue;
                    }
                    Symbol << " ";
                }

                if( XTemperaturePowerValue != 0 )
                {
                    Symbol << "Th";
                    if( XTemperaturePowerValue != 1 )
                    {
                        Symbol << "^" << XTemperaturePowerValue;
                    }
                    Symbol << " ";
                }

                if( XIntensityPowerValue != 0 )
                {
                    Symbol << "I";
                    if( XIntensityPowerValue != 1 )
                    {
                        Symbol << "^" << XIntensityPowerValue;
                    }
                    Symbol << " ";
                }

                if( XAnglePowerValue != 0 )
                {
                    Symbol << "A";
                    if( XAnglePowerValue != 1 )
                    {
                        Symbol << "^" << XAnglePowerValue;
                    }
                    Symbol << " ";
                }

                Symbol << "]";

                return Symbol.str();
            }

    };

    //eclipse cannot understand this line, but it is nonetheless correct.
    template< int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue, int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue >
    const string KDimension< KTypeList< KTypeInteger< XMassPowerValue >, KTypeList< KTypeInteger< XLengthPowerValue >, KTypeList< KTypeInteger< XTimePowerValue >, KTypeList< KTypeInteger< XChargePowerValue >, KTypeList< KTypeInteger< XTemperaturePowerValue >, KTypeList< KTypeInteger< XIntensityPowerValue >, KTypeList< KTypeInteger< XAnglePowerValue >, KSTypeNull > > > > > > > >::fSymbol = KDimension< KTypeList< KTypeInteger< XMassPowerValue >, KTypeList< KTypeInteger< XLengthPowerValue >, KTypeList< KTypeInteger< XTimePowerValue >, KTypeList< KTypeInteger< XChargePowerValue >, KTypeList< KTypeInteger< XTemperaturePowerValue >, KTypeList< KTypeInteger< XIntensityPowerValue >, KTypeList< KTypeInteger< XAnglePowerValue >, KSTypeNull > > > > > > > >::ConstructDimensionSymbol();

    //******************
    //dimension typedefs
    //******************

    //dimensionless                   //mass              //length            //time              //charge            //temperature       //intensity         //angle
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KDimensionless;

    //base dimensions
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSMassDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSLengthDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSTimeDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSChargeDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSTemperatureDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 > ) > KSIntensityDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 > ) > KSAngleDimension;

    //derived mechanical dimensions
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSAreaDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 3 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSVolumeDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSFrequencyDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 > ) > KSAngularFrequencyDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSVelocityDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSAccelerationDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSMomentumDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSForceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 > ) > KSAngularMomentumDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 > ) > KSTorqueDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSEnergyDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -3 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSPowerDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSPressureDimension;

    //derived electromagnetic dimensions
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -2 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSElectricPotentialDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -2 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSElectricFieldDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSMagneticPotentialDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSMagneticFieldDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< -1 >, KTypeInteger< -3 >, KTypeInteger< 2 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSElectricPermittivityDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSMagneticPermeabilityDimension;

    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSCurrentDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -1 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSResistanceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSInductanceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< -1 >, KTypeInteger< -2 >, KTypeInteger< 2 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSCapacitanceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KSMagneticFluxDimension;

}

#endif

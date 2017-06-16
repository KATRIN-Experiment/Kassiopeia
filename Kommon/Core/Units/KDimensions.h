#ifndef KSDIMENSIONS_H_
#define KSDIMENSIONS_H_

#include "KTypeList.h"
#include "KTypeInteger.h"
#include "KTypeLogic.h"

#include <string>
#include <sstream>

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
                        Value = KTypeEqual< KDimension< KTYPELIST7( KTypeInteger< XMassPowerValue >, KTypeInteger< XLengthPowerValue >, KTypeInteger< XTimePowerValue >, KTypeInteger< XChargePowerValue >, KTypeInteger< XTemperaturePowerValue >, KTypeInteger< XIntensityPowerValue >, KTypeInteger< XAnglePowerValue > ) >, KDimension< KTYPELIST7( KTypeInteger< XRightMassPowerValue >, KTypeInteger< XRightLengthPowerValue >, KTypeInteger< XRightTimePowerValue >, KTypeInteger< XRightChargePowerValue >, KTypeInteger< XRightTemperaturePowerValue >, KTypeInteger< XRightIntensityPowerValue >, KTypeInteger< XRightAnglePowerValue > ) > >::Value
                    };
            };

        public:
            static const std::string fSymbol;
            static const std::string ConstructDimensionSymbol()
            {

                std::stringstream Symbol;
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
    const std::string KDimension< KTypeList< KTypeInteger< XMassPowerValue >, KTypeList< KTypeInteger< XLengthPowerValue >, KTypeList< KTypeInteger< XTimePowerValue >, KTypeList< KTypeInteger< XChargePowerValue >, KTypeList< KTypeInteger< XTemperaturePowerValue >, KTypeList< KTypeInteger< XIntensityPowerValue >, KTypeList< KTypeInteger< XAnglePowerValue >, KTypeNull > > > > > > > >::fSymbol = KDimension< KTypeList< KTypeInteger< XMassPowerValue >, KTypeList< KTypeInteger< XLengthPowerValue >, KTypeList< KTypeInteger< XTimePowerValue >, KTypeList< KTypeInteger< XChargePowerValue >, KTypeList< KTypeInteger< XTemperaturePowerValue >, KTypeList< KTypeInteger< XIntensityPowerValue >, KTypeList< KTypeInteger< XAnglePowerValue >, KTypeNull > > > > > > > >::ConstructDimensionSymbol();

    //******************
    //dimension typedefs
    //******************

    //dimensionless                   //mass              //length            //time              //charge            //temperature       //intensity         //angle
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KDimensionless;

    //base dimensions
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KMassDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KLengthDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KTimeDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KChargeDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KTemperatureDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< 0 > ) > KIntensityDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 > ) > KAngleDimension;

    //derived mechanical dimensions
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KAreaDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 3 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KVolumeDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KFrequencyDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 1 > ) > KAngularFrequencyDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KVelocityDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 1 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KAccelerationDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KMomentumDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KForceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 > ) > KAngularMomentumDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 > ) > KTorqueDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KEnergyDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -3 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KPowerDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KPressureDimension;

    //derived electromagnetic dimensions
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -2 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KElectricPotentialDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -2 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KElectricFieldDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< -1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KMagneticPotentialDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KMagneticFieldDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< -1 >, KTypeInteger< -3 >, KTypeInteger< 2 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KElectricPermittivityDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< -2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KMagneticPermeabilityDimension;

    typedef KDimension< KTYPELIST7( KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KCurrentDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -1 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KResistanceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KInductanceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< -1 >, KTypeInteger< -2 >, KTypeInteger< 2 >, KTypeInteger< 2 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KCapacitanceDimension;
    typedef KDimension< KTYPELIST7( KTypeInteger< 1 >, KTypeInteger< 2 >, KTypeInteger< -1 >, KTypeInteger< -1 >, KTypeInteger< 0 >, KTypeInteger< 0 >, KTypeInteger< 0 > ) > KMagneticFluxDimension;

}

#endif

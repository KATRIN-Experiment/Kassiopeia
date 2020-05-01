#ifndef KUNITS_H_
#define KUNITS_H_

#include "KDimensions.h"

namespace katrin
{

//***************
//unit definition
//***************

template<class XDimensionType> class KUnit;

template<int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue,
         int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue>
class KUnit<
    KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>, KTypeInteger<XTimePowerValue>,
                          KTypeInteger<XChargePowerValue>, KTypeInteger<XTemperaturePowerValue>,
                          KTypeInteger<XIntensityPowerValue>, KTypeInteger<XAnglePowerValue>)>>
{
  public:
    typedef KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>,
                                  KTypeInteger<XTimePowerValue>, KTypeInteger<XChargePowerValue>,
                                  KTypeInteger<XTemperaturePowerValue>, KTypeInteger<XIntensityPowerValue>,
                                  KTypeInteger<XAnglePowerValue>)>
        Dimension;

  public:
    static const std::string fSymbol;
    static const std::string ConstructUnitSymbol()
    {

        std::stringstream Symbol;
        Symbol.clear();
        Symbol.str("");

        Symbol << "[ ";

        if (XMassPowerValue != 0) {
            Symbol << "kg";
            if (XMassPowerValue != 1) {
                Symbol << "^" << XMassPowerValue;
            }
            Symbol << " ";
        }

        if (XLengthPowerValue != 0) {
            Symbol << "m";
            if (XLengthPowerValue != 1) {
                Symbol << "^" << XLengthPowerValue;
            }
            Symbol << " ";
        }

        if (XTimePowerValue != 0) {
            Symbol << "s";
            if (XTimePowerValue != 1) {
                Symbol << "^" << XTimePowerValue;
            }
            Symbol << " ";
        }

        if (XChargePowerValue != 0) {
            Symbol << "C";
            if (XChargePowerValue != 1) {
                Symbol << "^" << XChargePowerValue;
            }
            Symbol << " ";
        }

        if (XTemperaturePowerValue != 0) {
            Symbol << "K";
            if (XTemperaturePowerValue != 1) {
                Symbol << "^" << XTemperaturePowerValue;
            }
            Symbol << " ";
        }

        if (XIntensityPowerValue != 0) {
            Symbol << "cd";
            if (XIntensityPowerValue != 1) {
                Symbol << "^" << XIntensityPowerValue;
            }
            Symbol << " ";
        }

        if (XAnglePowerValue != 0) {
            Symbol << "rad";
            if (XAnglePowerValue != 1) {
                Symbol << "^" << XAnglePowerValue;
            }
            Symbol << " ";
        }

        Symbol << "]";

        return Symbol.str();
    }
};

//can't use macros on this line because eclipse thinks it's an error
template<int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue,
         int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue>
const std::string KUnit<KDimension<KTypeList<
    KTypeInteger<XMassPowerValue>,
    KTypeList<KTypeInteger<XLengthPowerValue>,
              KTypeList<KTypeInteger<XTimePowerValue>,
                        KTypeList<KTypeInteger<XChargePowerValue>,
                                  KTypeList<KTypeInteger<XTemperaturePowerValue>,
                                            KTypeList<KTypeInteger<XIntensityPowerValue>,
                                                      KTypeList<KTypeInteger<XAnglePowerValue>, KTypeNull>>>>>>>>>::
    fSymbol = KUnit<KDimension<KTypeList<
        KTypeInteger<XMassPowerValue>,
        KTypeList<KTypeInteger<XLengthPowerValue>,
                  KTypeList<KTypeInteger<XTimePowerValue>,
                            KTypeList<KTypeInteger<XChargePowerValue>,
                                      KTypeList<KTypeInteger<XTemperaturePowerValue>,
                                                KTypeList<KTypeInteger<XIntensityPowerValue>,
                                                          KTypeList<KTypeInteger<XAnglePowerValue>, KTypeNull>>>>>>>>>::
        ConstructUnitSymbol();

//**********************
//scaled unit definition
//**********************

template<class XUnitType, class XTag> class KScaledUnit;

template<int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue,
         int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue, class XTag>
class KScaledUnit<
    KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>, KTypeInteger<XTimePowerValue>,
                          KTypeInteger<XChargePowerValue>, KTypeInteger<XTemperaturePowerValue>,
                          KTypeInteger<XIntensityPowerValue>, KTypeInteger<XAnglePowerValue>)>,
    XTag>
{
  public:
    typedef KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>,
                                  KTypeInteger<XTimePowerValue>, KTypeInteger<XChargePowerValue>,
                                  KTypeInteger<XTemperaturePowerValue>, KTypeInteger<XIntensityPowerValue>,
                                  KTypeInteger<XAnglePowerValue>)>
        Dimension;

  public:
    static const std::string fSymbol;
    static const double fScaleToThisUnitFromBaseUnit;
};

//**********************
//offset unit definition
//**********************

template<class XUnitType, class XTag> class KOffsetUnit;

template<int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue,
         int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue, class XTag>
class KOffsetUnit<
    KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>, KTypeInteger<XTimePowerValue>,
                          KTypeInteger<XChargePowerValue>, KTypeInteger<XTemperaturePowerValue>,
                          KTypeInteger<XIntensityPowerValue>, KTypeInteger<XAnglePowerValue>)>,
    XTag>
{
  public:
    typedef KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>,
                                  KTypeInteger<XTimePowerValue>, KTypeInteger<XChargePowerValue>,
                                  KTypeInteger<XTemperaturePowerValue>, KTypeInteger<XIntensityPowerValue>,
                                  KTypeInteger<XAnglePowerValue>)>
        Dimension;

  public:
    static const std::string fSymbol;
    static const double fOffsetToThisUnitFromBaseUnit;
};

//**************
//units typedefs
//**************

//unitless
typedef KUnit<KDimensionless> KUnitless;

//base units
typedef KUnit<KMassDimension> KKilogram;
typedef KUnit<KLengthDimension> KMeter;
typedef KUnit<KTimeDimension> KSecond;
typedef KUnit<KChargeDimension> KCoulomb;
typedef KUnit<KTemperatureDimension> KKelvin;
typedef KUnit<KIntensityDimension> KCandela;
typedef KUnit<KAngleDimension> KRadian;

//derived units
typedef KUnit<KAreaDimension> KSquareMeter;
typedef KUnit<KVolumeDimension> KCubicMeter;
typedef KUnit<KFrequencyDimension> KHertz;
typedef KUnit<KAngularFrequencyDimension> KRadianPerSecond;
typedef KUnit<KVelocityDimension> KMeterPerSecond;
typedef KUnit<KAccelerationDimension> KMeterPerSecondSquared;
typedef KUnit<KMomentumDimension> KKilogramMeterPerSecond;
typedef KUnit<KForceDimension> KNewton;
typedef KUnit<KAngularMomentumDimension> KKilogramMeterSquaredPerSecondPerRadian;
typedef KUnit<KTorqueDimension> KKilogramMeterSquaredPerSecondSquaredPerRadian;
typedef KUnit<KEnergyDimension> KJoule;
typedef KUnit<KPowerDimension> KJoulePerSecond;

typedef KUnit<KElectricPotentialDimension> KVolt;
typedef KUnit<KElectricFieldDimension> KVoltPerMeter;
typedef KUnit<KMagneticPotentialDimension> KTeslaMeter;
typedef KUnit<KMagneticFieldDimension> KTesla;
typedef KUnit<KElectricPermittivityDimension> KFaradPerMeter;
typedef KUnit<KMagneticPermeabilityDimension> KHenryPerMeter;

typedef KUnit<KCurrentDimension> KAmpere;
typedef KUnit<KResistanceDimension> KOhm;
typedef KUnit<KInductanceDimension> KHenry;
typedef KUnit<KCapacitanceDimension> KFarad;
typedef KUnit<KMagneticFluxDimension> KWeber;

//scaled units
class KLiterTag;
typedef KScaledUnit<KVolumeDimension, KLiterTag> KLiter;

class KElectronVoltTag;
typedef KScaledUnit<KEnergyDimension, KElectronVoltTag> KElectronVolt;

class KGaussTag;
typedef KScaledUnit<KMagneticFieldDimension, KGaussTag> KGauss;

class KDegreeTag;
typedef KScaledUnit<KAngleDimension, KDegreeTag> KDegree;

//offset units
class KCelsiusTag;
typedef KOffsetUnit<KTemperatureDimension, KCelsiusTag> KCelsius;

}  // namespace katrin

#endif

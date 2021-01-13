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
    using Dimension = KDimension<::katrin::KTypeList<
        KTypeInteger<XMassPowerValue>,
        ::katrin::KTypeList<
            KTypeInteger<XLengthPowerValue>,
            ::katrin::KTypeList<
                KTypeInteger<XTimePowerValue>,
                ::katrin::KTypeList<
                    KTypeInteger<XChargePowerValue>,
                    ::katrin::KTypeList<
                        KTypeInteger<XTemperaturePowerValue>,
                        ::katrin::KTypeList<KTypeInteger<XIntensityPowerValue>,
                                            ::katrin::KTypeList<KTypeInteger<XAnglePowerValue>, KTypeNull>>>>>>>>;

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
    using Dimension = KDimension<::katrin::KTypeList<
        KTypeInteger<XMassPowerValue>,
        ::katrin::KTypeList<
            KTypeInteger<XLengthPowerValue>,
            ::katrin::KTypeList<
                KTypeInteger<XTimePowerValue>,
                ::katrin::KTypeList<
                    KTypeInteger<XChargePowerValue>,
                    ::katrin::KTypeList<
                        KTypeInteger<XTemperaturePowerValue>,
                        ::katrin::KTypeList<KTypeInteger<XIntensityPowerValue>,
                                            ::katrin::KTypeList<KTypeInteger<XAnglePowerValue>, KTypeNull>>>>>>>>;

  public:
    static const std::string fSymbol;
    static const double fOffsetToThisUnitFromBaseUnit;
};

//**************
//units typedefs
//**************

//unitless
using KUnitless = KUnit<KDimensionless>;

//base units
using KKilogram = KUnit<KMassDimension>;
using KMeter = KUnit<KLengthDimension>;
using KSecond = KUnit<KTimeDimension>;
using KCoulomb = KUnit<KChargeDimension>;
using KKelvin = KUnit<KTemperatureDimension>;
using KCandela = KUnit<KIntensityDimension>;
using KRadian = KUnit<KAngleDimension>;

//derived units
using KSquareMeter = KUnit<KAreaDimension>;
using KCubicMeter = KUnit<KVolumeDimension>;
using KHertz = KUnit<KFrequencyDimension>;
using KRadianPerSecond = KUnit<KAngularFrequencyDimension>;
using KMeterPerSecond = KUnit<KVelocityDimension>;
using KMeterPerSecondSquared = KUnit<KAccelerationDimension>;
using KKilogramMeterPerSecond = KUnit<KMomentumDimension>;
using KNewton = KUnit<KForceDimension>;
using KKilogramMeterSquaredPerSecondPerRadian = KUnit<KAngularMomentumDimension>;
using KKilogramMeterSquaredPerSecondSquaredPerRadian = KUnit<KTorqueDimension>;
using KJoule = KUnit<KEnergyDimension>;
using KJoulePerSecond = KUnit<KPowerDimension>;

using KVolt = KUnit<KElectricPotentialDimension>;
using KVoltPerMeter = KUnit<KElectricFieldDimension>;
using KTeslaMeter = KUnit<KMagneticPotentialDimension>;
using KTesla = KUnit<KMagneticFieldDimension>;
using KFaradPerMeter = KUnit<KElectricPermittivityDimension>;
using KHenryPerMeter = KUnit<KMagneticPermeabilityDimension>;

using KAmpere = KUnit<KCurrentDimension>;
using KOhm = KUnit<KResistanceDimension>;
using KHenry = KUnit<KInductanceDimension>;
using KFarad = KUnit<KCapacitanceDimension>;
using KWeber = KUnit<KMagneticFluxDimension>;

//scaled units
class KLiterTag;
using KLiter = KScaledUnit<KVolumeDimension, KLiterTag>;

class KElectronVoltTag;
using KElectronVolt = KScaledUnit<KEnergyDimension, KElectronVoltTag>;

class KGaussTag;
using KGauss = KScaledUnit<KMagneticFieldDimension, KGaussTag>;

class KDegreeTag;
using KDegree = KScaledUnit<KAngleDimension, KDegreeTag>;

//offset units
class KCelsiusTag;
using KCelsius = KOffsetUnit<KTemperatureDimension, KCelsiusTag>;

}  // namespace katrin

#endif

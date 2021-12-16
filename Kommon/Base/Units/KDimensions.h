#ifndef KSDIMENSIONS_H_
#define KSDIMENSIONS_H_

#include "KTypeInteger.h"
#include "KTypeList.h"
#include "KTypeLogic.h"

#include <sstream>
#include <string>

namespace katrin
{

//********************
//dimension definition
//********************

template<class XDimensionList> class KDimension;

template<int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue,
         int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue>
class KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>,
                            KTypeInteger<XTimePowerValue>, KTypeInteger<XChargePowerValue>,
                            KTypeInteger<XTemperaturePowerValue>, KTypeInteger<XIntensityPowerValue>,
                            KTypeInteger<XAnglePowerValue>)>
{
  public:
    template<class XRight> class Multiply;

    template<int XRightMassPowerValue, int XRightLengthPowerValue, int XRightTimePowerValue, int XRightChargePowerValue,
             int XRightTemperaturePowerValue, int XRightIntensityPowerValue, int XRightAnglePowerValue>
    class Multiply<KDimension<KTYPELIST7(KTypeInteger<XRightMassPowerValue>, KTypeInteger<XRightLengthPowerValue>,
                                         KTypeInteger<XRightTimePowerValue>, KTypeInteger<XRightChargePowerValue>,
                                         KTypeInteger<XRightTemperaturePowerValue>,
                                         KTypeInteger<XRightIntensityPowerValue>, KTypeInteger<XRightAnglePowerValue>)>>
    {
      public:
        typedef KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue + XRightMassPowerValue>,
                                      KTypeInteger<XLengthPowerValue + XRightLengthPowerValue>,
                                      KTypeInteger<XTimePowerValue + XRightTimePowerValue>,
                                      KTypeInteger<XChargePowerValue + XRightChargePowerValue>,
                                      KTypeInteger<XTemperaturePowerValue + XRightTemperaturePowerValue>,
                                      KTypeInteger<XIntensityPowerValue + XRightIntensityPowerValue>,
                                      KTypeInteger<XAnglePowerValue + XRightAnglePowerValue>)>
            Type;
    };

    template<class XRight> class Divide;

    template<int XRightMassPowerValue, int XRightLengthPowerValue, int XRightTimePowerValue, int XRightChargePowerValue,
             int XRightTemperaturePowerValue, int XRightIntensityPowerValue, int XRightAnglePowerValue>
    class Divide<KDimension<KTYPELIST7(KTypeInteger<XRightMassPowerValue>, KTypeInteger<XRightLengthPowerValue>,
                                       KTypeInteger<XRightTimePowerValue>, KTypeInteger<XRightChargePowerValue>,
                                       KTypeInteger<XRightTemperaturePowerValue>,
                                       KTypeInteger<XRightIntensityPowerValue>, KTypeInteger<XRightAnglePowerValue>)>>
    {
      public:
        using Type = KDimension<::katrin::KTypeList<
            KTypeInteger<XMassPowerValue - XRightMassPowerValue>,
            ::katrin::KTypeList<
                KTypeInteger<XLengthPowerValue - XRightLengthPowerValue>,
                ::katrin::KTypeList<
                    KTypeInteger<XTimePowerValue - XRightTimePowerValue>,
                    ::katrin::KTypeList<
                        KTypeInteger<XChargePowerValue - XRightChargePowerValue>,
                        ::katrin::KTypeList<
                            KTypeInteger<XTemperaturePowerValue - XRightTemperaturePowerValue>,
                            ::katrin::KTypeList<
                                KTypeInteger<XIntensityPowerValue - XRightIntensityPowerValue>,
                                ::katrin::KTypeList<KTypeInteger<XAnglePowerValue - XRightAnglePowerValue>,
                                    KTypeNull>>>>>>>>;
    };

    template<class XRight> class Equal;

    template<int XRightMassPowerValue, int XRightLengthPowerValue, int XRightTimePowerValue, int XRightChargePowerValue,
             int XRightTemperaturePowerValue, int XRightIntensityPowerValue, int XRightAnglePowerValue>
    class Equal<KDimension<KTYPELIST7(KTypeInteger<XRightMassPowerValue>, KTypeInteger<XRightLengthPowerValue>,
                                      KTypeInteger<XRightTimePowerValue>, KTypeInteger<XRightChargePowerValue>,
                                      KTypeInteger<XRightTemperaturePowerValue>,
                                      KTypeInteger<XRightIntensityPowerValue>, KTypeInteger<XRightAnglePowerValue>)>>
    {
      public:
        enum
        {
            Value =
                KTypeEqual<KDimension<KTYPELIST7(KTypeInteger<XMassPowerValue>, KTypeInteger<XLengthPowerValue>,
                                                 KTypeInteger<XTimePowerValue>, KTypeInteger<XChargePowerValue>,
                                                 KTypeInteger<XTemperaturePowerValue>,
                                                 KTypeInteger<XIntensityPowerValue>, KTypeInteger<XAnglePowerValue>)>,
                           KDimension<KTYPELIST7(
                               KTypeInteger<XRightMassPowerValue>, KTypeInteger<XRightLengthPowerValue>,
                               KTypeInteger<XRightTimePowerValue>, KTypeInteger<XRightChargePowerValue>,
                               KTypeInteger<XRightTemperaturePowerValue>, KTypeInteger<XRightIntensityPowerValue>,
                               KTypeInteger<XRightAnglePowerValue>)>>::Value
        };
    };

  public:
    static const std::string fSymbol;
    static const std::string ConstructDimensionSymbol()
    {

        std::stringstream Symbol;
        Symbol.clear();
        Symbol.str("");

        Symbol << "[ ";

        if (XMassPowerValue != 0) {
            Symbol << "M";
            if (XMassPowerValue != 1) {
                Symbol << "^" << XMassPowerValue;
            }
            Symbol << " ";
        }

        if (XLengthPowerValue != 0) {
            Symbol << "L";
            if (XLengthPowerValue != 1) {
                Symbol << "^" << XLengthPowerValue;
            }
            Symbol << " ";
        }

        if (XTimePowerValue != 0) {
            Symbol << "T";
            if (XTimePowerValue != 1) {
                Symbol << "^" << XTimePowerValue;
            }
            Symbol << " ";
        }

        if (XChargePowerValue != 0) {
            Symbol << "Q";
            if (XChargePowerValue != 1) {
                Symbol << "^" << XChargePowerValue;
            }
            Symbol << " ";
        }

        if (XTemperaturePowerValue != 0) {
            Symbol << "Th";
            if (XTemperaturePowerValue != 1) {
                Symbol << "^" << XTemperaturePowerValue;
            }
            Symbol << " ";
        }

        if (XIntensityPowerValue != 0) {
            Symbol << "I";
            if (XIntensityPowerValue != 1) {
                Symbol << "^" << XIntensityPowerValue;
            }
            Symbol << " ";
        }

        if (XAnglePowerValue != 0) {
            Symbol << "A";
            if (XAnglePowerValue != 1) {
                Symbol << "^" << XAnglePowerValue;
            }
            Symbol << " ";
        }

        Symbol << "]";

        return Symbol.str();
    }
};

//eclipse cannot understand this line, but it is nonetheless correct.
template<int XMassPowerValue, int XLengthPowerValue, int XTimePowerValue, int XChargePowerValue,
         int XTemperaturePowerValue, int XIntensityPowerValue, int XAnglePowerValue>
const std::string KDimension<KTypeList<
    KTypeInteger<XMassPowerValue>,
    KTypeList<KTypeInteger<XLengthPowerValue>,
              KTypeList<KTypeInteger<XTimePowerValue>,
                        KTypeList<KTypeInteger<XChargePowerValue>,
                                  KTypeList<KTypeInteger<XTemperaturePowerValue>,
                                            KTypeList<KTypeInteger<XIntensityPowerValue>,
                                                      KTypeList<KTypeInteger<XAnglePowerValue>, KTypeNull>>>>>>>>::
    fSymbol = KDimension<KTypeList<
        KTypeInteger<XMassPowerValue>,
        KTypeList<KTypeInteger<XLengthPowerValue>,
                  KTypeList<KTypeInteger<XTimePowerValue>,
                            KTypeList<KTypeInteger<XChargePowerValue>,
                                      KTypeList<KTypeInteger<XTemperaturePowerValue>,
                                                KTypeList<KTypeInteger<XIntensityPowerValue>,
                                                          KTypeList<KTypeInteger<XAnglePowerValue>, KTypeNull>>>>>>>>::
        ConstructDimensionSymbol();

//******************
//dimension typedefs
//******************

//dimensionless                   //mass              //length            //time              //charge            //temperature       //intensity         //angle
using KDimensionless = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;

//base dimensions
using KMassDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KLengthDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KTimeDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<1>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KChargeDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KTemperatureDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<1>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KIntensityDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<1>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KAngleDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<1>, KTypeNull>>>>>>>>;

//derived mechanical dimensions
using KAreaDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KVolumeDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<3>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KFrequencyDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KAngularFrequencyDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<1>, KTypeNull>>>>>>>>;
using KVelocityDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KAccelerationDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<-2>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KMomentumDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KForceDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<-2>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KAngularMomentumDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<-1>, KTypeNull>>>>>>>>;
using KTorqueDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<-2>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<-1>, KTypeNull>>>>>>>>;
using KEnergyDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<-2>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KPowerDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<-3>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KPressureDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<-1>,
        ::katrin::KTypeList<
            KTypeInteger<-2>,
            ::katrin::KTypeList<
                KTypeInteger<0>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;

//derived electromagnetic dimensions
using KElectricPotentialDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<-2>,
            ::katrin::KTypeList<
                KTypeInteger<-1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KElectricFieldDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<-2>,
            ::katrin::KTypeList<
                KTypeInteger<-1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KMagneticPotentialDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<-1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KMagneticFieldDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<-1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KElectricPermittivityDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<-1>,
    ::katrin::KTypeList<
        KTypeInteger<-3>,
        ::katrin::KTypeList<
            KTypeInteger<2>,
            ::katrin::KTypeList<
                KTypeInteger<2>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KMagneticPermeabilityDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<1>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<-2>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;

using KCurrentDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<0>,
    ::katrin::KTypeList<
        KTypeInteger<0>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KResistanceDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<2>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KInductanceDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<0>,
            ::katrin::KTypeList<
                KTypeInteger<-1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KCapacitanceDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<-1>,
    ::katrin::KTypeList<
        KTypeInteger<-2>,
        ::katrin::KTypeList<
            KTypeInteger<2>,
            ::katrin::KTypeList<
                KTypeInteger<2>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;
using KMagneticFluxDimension = KDimension<::katrin::KTypeList<
    KTypeInteger<1>,
    ::katrin::KTypeList<
        KTypeInteger<2>,
        ::katrin::KTypeList<
            KTypeInteger<-1>,
            ::katrin::KTypeList<
                KTypeInteger<-1>,
                ::katrin::KTypeList<
                    KTypeInteger<0>,
                    ::katrin::KTypeList<KTypeInteger<0>, ::katrin::KTypeList<KTypeInteger<0>, KTypeNull>>>>>>>>;

}  // namespace katrin

#endif

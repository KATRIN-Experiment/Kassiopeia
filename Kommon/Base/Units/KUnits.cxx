#include "KUnits.h"

using namespace std;

namespace katrin
{
//special symbols
template<> const string KHertz::fSymbol = string("[ Hz ]");
template<> const string KNewton::fSymbol = string("[ N ]");
template<> const string KJoule::fSymbol = string("[ J ]");
template<> const string KJoulePerSecond::fSymbol = string("[ J/s ]");

template<> const string KVolt::fSymbol = string("[ V ]");
template<> const string KVoltPerMeter::fSymbol = string("[ V/m ]");
template<> const string KTeslaMeter::fSymbol = string("[ T*m ]");
template<> const string KTesla::fSymbol = string("[ T ]");
template<> const string KFaradPerMeter::fSymbol = string("[ F/m ]");
template<> const string KHenryPerMeter::fSymbol = string("[ H/m ]");

template<> const string KAmpere::fSymbol = string("[ A ]");
template<> const string KOhm::fSymbol = string("[ Ohm ]");
template<> const string KHenry::fSymbol = string("[ H ]");
template<> const string KFarad::fSymbol = string("[ F ]");
template<> const string KWeber::fSymbol = string("[ Wb ]");

//scaled units
template<> const string KLiter::fSymbol = string("[ L ]");
template<> const double KLiter::fScaleToThisUnitFromBaseUnit = 1000.;  //1 m^3 = 1 000 L

template<> const string KElectronVolt::fSymbol = string("[ eV ]");
template<> const double KElectronVolt::fScaleToThisUnitFromBaseUnit = 6.24150974e18;  //1 J = 6.24...x 10^18 eV

template<> const string KGauss::fSymbol = string("[ G ]");
template<> const double KGauss::fScaleToThisUnitFromBaseUnit = 10000.;  //1 T = 10 000 G

template<> const string KDegree::fSymbol = string("[ deg ]");
template<> const double KDegree::fScaleToThisUnitFromBaseUnit = 57.2957795;  //1 rad = 57.3 deg

//offset units
template<> const string KCelsius::fSymbol = string("[ C ]");
template<> const double KCelsius::fOffsetToThisUnitFromBaseUnit = -273.15;  //0 K = -273.15 C
}  // namespace katrin

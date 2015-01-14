#include "KUnits.h"

namespace katrin
{
    //special symbols
    template< > const string KSHertz::fSymbol = string( "[ Hz ]" );
    template< > const string KSNewton::fSymbol = string( "[ N ]" );
    template< > const string KSJoule::fSymbol = string( "[ J ]" );
    template< > const string KSJoulePerSecond::fSymbol = string( "[ J/s ]" );

    template< > const string KSVolt::fSymbol = string( "[ V ]" );
    template< > const string KSVoltPerMeter::fSymbol = string( "[ V/m ]" );
    template< > const string KSTeslaMeter::fSymbol = string( "[ T*m ]" );
    template< > const string KSTesla::fSymbol = string( "[ T ]" );
    template< > const string KSFaradPerMeter::fSymbol = string( "[ F/m ]" );
    template< > const string KSHenryPerMeter::fSymbol = string( "[ H/m ]" );

    template< > const string KSAmpere::fSymbol = string( "[ A ]" );
    template< > const string KSOhm::fSymbol = string( "[ Ohm ]" );
    template< > const string KSHenry::fSymbol = string( "[ H ]" );
    template< > const string KSFarad::fSymbol = string( "[ F ]" );
    template< > const string KSWeber::fSymbol = string( "[ Wb ]" );

    //scaled units
    template< > const string KSLiter::fSymbol = string( "[ L ]" );
    template< > const double KSLiter::fScaleToThisUnitFromBaseUnit = 1000.; //1 m^3 = 1 000 L

    template< > const string KSElectronVolt::fSymbol = string( "[ eV ]" );
    template< > const double KSElectronVolt::fScaleToThisUnitFromBaseUnit = 6.24150974e18; //1 J = 6.24...x 10^18 eV

    template< > const string KSGauss::fSymbol = string( "[ G ]" );
    template< > const double KSGauss::fScaleToThisUnitFromBaseUnit = 10000.; //1 T = 10 000 G

    template< > const string KSDegree::fSymbol = string( "[ deg ]" );
    template< > const double KSDegree::fScaleToThisUnitFromBaseUnit = 57.2957795; //1 rad = 57.3 deg

    //offset units
    template< > const string KSCelsius::fSymbol = string( "[ C ]" );
    template< > const double KSCelsius::fOffsetToThisUnitFromBaseUnit = -273.15; //0 K = -273.15 C
}


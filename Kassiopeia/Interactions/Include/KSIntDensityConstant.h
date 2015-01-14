#ifndef Kassiopeia_KSIntDensityConstant_h_
#define Kassiopeia_KSIntDensityConstant_h_

#include "KSIntDensity.h"

#include "KField.h"

namespace Kassiopeia
{
    class KSIntDensityConstant :
        public KSComponentTemplate< KSIntDensityConstant, KSIntDensity >
    {
        public:
            KSIntDensityConstant();
            KSIntDensityConstant( const KSIntDensityConstant& aCopy );
            KSIntDensityConstant* Clone() const;
            virtual ~KSIntDensityConstant();

        public:
            void CalculateDensity( const KSParticle& aParticle, double& aDensity );

        public:
            K_SET_GET( double, Temperature ); // kelvin
            K_SET_GET( double, Pressure ); // pascal (SI UNITS!)
            K_SET_GET( double, Density );  // m^-3
    };

}

#endif

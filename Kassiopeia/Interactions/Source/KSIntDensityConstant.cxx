#include "KSIntDensityConstant.h"
#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSIntDensityConstant::KSIntDensityConstant() :
        fTemperature( 0. ),
        fPressure( 0. ),
        fDensity( 0. )
    {
    }
    KSIntDensityConstant::KSIntDensityConstant( const KSIntDensityConstant& aCopy ) :
        fTemperature( aCopy.fTemperature ),
        fPressure( aCopy.fPressure ),
        fDensity( aCopy.fDensity )
    {
    }
    KSIntDensityConstant* KSIntDensityConstant::Clone() const
    {
        return new KSIntDensityConstant( *this );
    }
    KSIntDensityConstant::~KSIntDensityConstant()
    {
    }

    void KSIntDensityConstant::CalculateDensity( const KSParticle&, double& aDensity )
    {
        if( fDensity == 0. )
        {
            aDensity = fPressure / (KConst::kB() * fTemperature );
            return;
        }
        else
        {
            aDensity = fDensity;
            return;
        }
    }

}

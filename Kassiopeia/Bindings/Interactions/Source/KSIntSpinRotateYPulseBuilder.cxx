#include "KSIntSpinRotateYPulseBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSIntSpinRotateYPulseBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntSpinRotateYPulseStructure =
        KSIntSpinRotateYPulseBuilder::Attribute< string >( "name" ) +
        KSIntSpinRotateYPulseBuilder::Attribute< double >( "time" ) +
        KSIntSpinRotateYPulseBuilder::Attribute< double >( "angle" ) +
        KSIntSpinRotateYPulseBuilder::Attribute< bool >( "is_adiabatic" );

    STATICINT sKSIntSpinRotateYPulse =
        KSRootBuilder::ComplexElement< KSIntSpinRotateYPulse >( "ksint_spin_rotate_y_pulse" );

}

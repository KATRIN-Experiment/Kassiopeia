#include "KSIntSpinFlipPulseBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSIntSpinFlipPulseBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntSpinFlipPulseStructure =
        KSIntSpinFlipPulseBuilder::Attribute< string >( "name" ) +
        KSIntSpinFlipPulseBuilder::Attribute< double >( "time" );

    STATICINT sKSIntSpinFlipPulse =
        KSRootBuilder::ComplexElement< KSIntSpinFlipPulse >( "ksint_spin_flip_pulse" );

}

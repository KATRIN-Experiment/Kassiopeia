#include "KSIntDensityConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntDensityConstantBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntDensityConstantStructure =
        KSIntDensityConstantBuilder::Attribute< string >( "name" ) +
        KSIntDensityConstantBuilder::Attribute< double >( "temperature" ) +
        KSIntDensityConstantBuilder::Attribute< double >( "pressure" ) +
        KSIntDensityConstantBuilder::Attribute< double >( "pressure_mbar" ) +
		KSIntDensityConstantBuilder::Attribute< double >( "density" );

    STATICINT sToolboxKSIntDensityConstant =
        KSRootBuilder::ComplexElement< KSIntDensityConstant >( "ksint_density_constant" );
}

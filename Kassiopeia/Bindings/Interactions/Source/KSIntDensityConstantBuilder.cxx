#include "KSIntDensityConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntDensityConstantBuilder::~KComplexElement()
    {
    }

    static int sKSIntDensityConstantStructure =
        KSIntDensityConstantBuilder::Attribute< string >( "name" ) +
        KSIntDensityConstantBuilder::Attribute< double >( "temperature" ) +
        KSIntDensityConstantBuilder::Attribute< double >( "pressure" ) +
        KSIntDensityConstantBuilder::Attribute< double >( "pressure_mbar" ) +
		KSIntDensityConstantBuilder::Attribute< double >( "density" );

    static int sToolboxKSIntDensityConstant =
        KSRootBuilder::ComplexElement< KSIntDensityConstant >( "ksint_density_constant" );
}

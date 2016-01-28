#include "KSTrajControlMagneticMomentBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajControlMagneticMomentBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajControlMagneticMomentStructure =
        KSTrajControlMagneticMomentBuilder::Attribute< string >( "name" ) +
        KSTrajControlMagneticMomentBuilder::Attribute< double >( "lower_limit" ) +
        KSTrajControlMagneticMomentBuilder::Attribute< double >( "upper_limit" );

    STATICINT sToolboxKSTrajControlMagneticMoment =
        KSRootBuilder::ComplexElement< KSTrajControlMagneticMoment >( "kstraj_control_magnetic_moment" );

}

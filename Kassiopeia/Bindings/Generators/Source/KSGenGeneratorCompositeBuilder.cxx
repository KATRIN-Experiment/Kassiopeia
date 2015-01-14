#include "KSGenGeneratorCompositeBuilder.h"
#include "KSGenEnergyCompositeBuilder.h"
#include "KSGenEnergyKryptonEventBuilder.h"
#include "KSGenEnergyRadonEventBuilder.h"
#include "KSGenEnergyLeadEventBuilder.h"
#include "KSGenPositionRectangularCompositeBuilder.h"
#include "KSGenPositionCylindricalCompositeBuilder.h"
#include "KSGenPositionSphericalCompositeBuilder.h"
#include "KSGenPositionSpaceRandomBuilder.h"
#include "KSGenPositionSurfaceRandomBuilder.h"
#include "KSGenMomentumRectangularCompositeBuilder.h"
#include "KSGenDirectionSphericalCompositeBuilder.h"
#include "KSGenDirectionSurfaceCompositeBuilder.h"
#include "KSGenTimeCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenGeneratorCompositeBuilder::~KComplexElement()
    {
    }

    static int sKSGenGeneratorCompositeStructure =
        KSGenGeneratorCompositeBuilder::Attribute< string >( "name" ) +
        KSGenGeneratorCompositeBuilder::Attribute< long long >( "pid" ) +
        KSGenGeneratorCompositeBuilder::Attribute< string >( "special" ) +
        KSGenGeneratorCompositeBuilder::Attribute< string >( "creator" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenEnergyComposite >( "energy_composite" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenEnergyKryptonEvent >( "energy_krypton_event" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenEnergyRadonEvent >( "energy_radon_event" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenEnergyLeadEvent >( "energy_lead_event" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionRectangularComposite >( "position_rectangular_composite" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionCylindricalComposite >( "position_cylindrical_composite" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionSphericalComposite >( "position_spherical_composite" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionSpaceRandom >( "position_space_random" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenPositionSurfaceRandom >( "position_surface_random" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenMomentumRectangularComposite >( "momentum_rectangular_composite" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenDirectionSphericalComposite >( "direction_spherical_composite" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenDirectionSurfaceComposite >( "direction_surface_composite" ) +
        KSGenGeneratorCompositeBuilder::ComplexElement< KSGenTimeComposite >( "time_composite" );

    static int sKSGenGeneratorComposite =
        KSRootBuilder::ComplexElement< KSGenGeneratorComposite >( "ksgen_generator_composite" );

}

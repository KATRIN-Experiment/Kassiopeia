/*
 * KSGenPositionMaskBuilder.cxx
 *
 *  Created on: 25.04.2015
 *      Author: J. Behrens
 */

#include "KSGenPositionMaskBuilder.h"
#include "KSGenPositionRectangularCompositeBuilder.h"
#include "KSGenPositionCylindricalCompositeBuilder.h"
#include "KSGenPositionSphericalCompositeBuilder.h"
#include "KSGenPositionSurfaceRandomBuilder.h"
#include "KSGenPositionSpaceRandomBuilder.h"
#include "KSGenPositionMeshSurfaceRandomBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    template<>
    KSGenPositionMaskBuilder::~KComplexElement() {}

    STATICINT sKSGenPositionMaskStructure =
            KSGenPositionMaskBuilder::Attribute<string>("name") +
            KSGenPositionMaskBuilder::Attribute<string>("spaces_allowed") +
            KSGenPositionMaskBuilder::Attribute<string>("spaces_forbidden") +
            KSGenPositionMaskBuilder::Attribute<unsigned int>("max_retries") +
            KSGenPositionMaskBuilder::ComplexElement< KSGenPositionRectangularComposite >( "position_rectangular_composite" ) +
            KSGenPositionMaskBuilder::ComplexElement< KSGenPositionCylindricalComposite >( "position_cylindrical_composite" ) +
            KSGenPositionMaskBuilder::ComplexElement< KSGenPositionSphericalComposite >( "position_spherical_composite" ) +
            KSGenPositionMaskBuilder::ComplexElement< KSGenPositionSurfaceRandom >( "position_surface_random" ) +
            KSGenPositionMaskBuilder::ComplexElement< KSGenPositionSpaceRandom >( "position_space_random" ) +
            KSGenPositionMaskBuilder::ComplexElement< KSGenPositionMeshSurfaceRandom >( "position_mesh_surface_random" );

    STATICINT sKSGenPositionMask =
            KSRootBuilder::ComplexElement<KSGenPositionMask>("ksgen_position_mask");

}

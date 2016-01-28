/*
 * KSGenPositionMeshSurfaceRandom.cxx
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#include "KSGenPositionMeshSurfaceRandomBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    template<>
    KSGenPositionMeshSurfaceRandomBuilder::~KComplexElement() {}


    STATICINT sKSGenPositionMeshSurfaceRandomStructure =
        KSGenPositionMeshSurfaceRandomBuilder::Attribute< string >( "name" ) +
        KSGenPositionMeshSurfaceRandomBuilder::Attribute< string >( "surfaces" );

    STATICINT sKSGenPositionMeshSurfaceRandom =
        KSRootBuilder::ComplexElement< KSGenPositionMeshSurfaceRandom >( "ksgen_position_mesh_surface_random" );

}

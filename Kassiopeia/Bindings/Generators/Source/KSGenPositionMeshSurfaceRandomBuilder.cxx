/*
 * KSGenPositionMeshSurfaceRandom.cxx
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#include "KSGenPositionMeshSurfaceRandomBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{
template<> KSGenPositionMeshSurfaceRandomBuilder::~KComplexElement() = default;


STATICINT sKSGenPositionMeshSurfaceRandomStructure =
    KSGenPositionMeshSurfaceRandomBuilder::Attribute<std::string>("name") +
    KSGenPositionMeshSurfaceRandomBuilder::Attribute<std::string>("surfaces");

STATICINT sKSGenPositionMeshSurfaceRandom =
    KSRootBuilder::ComplexElement<KSGenPositionMeshSurfaceRandom>("ksgen_position_mesh_surface_random");

}  // namespace katrin

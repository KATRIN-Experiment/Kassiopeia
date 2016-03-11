#include "KGMeshDeformerBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    template< >
    KGMeshDeformerBuilder::~KComplexElement()
    {
    }

    STATICINT sKGMeshDeformerStructure =
      KGMeshDeformerBuilder::Attribute< string >( "surfaces" ) +
      KGMeshDeformerBuilder::Attribute< string >( "spaces" );

    STATICINT sKGMeshDeformer =
      KGInterfaceBuilder::ComplexElement< KGMeshDeformer >( "mesh_deformer" );

}

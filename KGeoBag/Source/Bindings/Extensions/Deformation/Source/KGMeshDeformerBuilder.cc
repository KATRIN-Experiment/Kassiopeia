#include "KGMeshDeformerBuilder.hh"
#include "KGInterfaceBuilder.hh"

namespace katrin
{

    template< >
    KGMeshDeformerBuilder::~KComplexElement()
    {
    }

    static int sKGMeshDeformerStructure =
      KGMeshDeformerBuilder::Attribute< string >( "surfaces" ) +
      KGMeshDeformerBuilder::Attribute< string >( "spaces" );

    static const int sKGMeshDeformer =
      KGInterfaceBuilder::ComplexElement< KGMeshDeformer >( "mesh_deformer" );

}

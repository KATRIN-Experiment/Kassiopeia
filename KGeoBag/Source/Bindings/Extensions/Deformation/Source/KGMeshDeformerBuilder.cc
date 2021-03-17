#include "KGMeshDeformerBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

template<> KGMeshDeformerBuilder::~KComplexElement() = default;

STATICINT sKGMeshDeformerStructure =
    KGMeshDeformerBuilder::Attribute<std::string>("surfaces") + KGMeshDeformerBuilder::Attribute<std::string>("spaces");

STATICINT sKGMeshDeformer = KGInterfaceBuilder::ComplexElement<KGMeshDeformer>("mesh_deformer");

}  // namespace katrin

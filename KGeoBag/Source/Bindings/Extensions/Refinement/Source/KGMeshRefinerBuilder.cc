#include "KGMeshRefinerBuilder.hh"

#include "KGInterfaceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace katrin
{

template<> KGMeshRefinerBuilder::~KComplexElement() = default;

STATICINT sKGMeshRefinerStructure =
    KGMeshRefinerBuilder::Attribute<std::string>("surfaces") +
    KGMeshRefinerBuilder::Attribute<std::string>("spaces") +
    KGMeshRefinerBuilder::Attribute<unsigned int>("max_refinement_steps") +
    KGMeshRefinerBuilder::Attribute<double>("max_length") +
    KGMeshRefinerBuilder::Attribute<double>("max_area") +
    KGMeshRefinerBuilder::Attribute<double>("max_aspect_ratio")
    ;

STATICINT sKGMeshRefiner = KGInterfaceBuilder::ComplexElement<KGMeshRefiner>("mesh_refiner");

}  // namespace katrin

#include "KGDiscreteRotationalMeshBuilder.hh"

#include "KGDiscreteRotationalMesher.hh"

using namespace std;
using namespace KGeoBag;

namespace KGeoBag
{

KGDiscreteRotationalMeshAttributor::KGDiscreteRotationalMeshAttributor() : fAxialAngle(0.), fAxialCount(100) {}

KGDiscreteRotationalMeshAttributor::~KGDiscreteRotationalMeshAttributor()
{
    KGDiscreteRotationalMesher tMesher;
    tMesher.SetAxialAngle(fAxialAngle);
    tMesher.SetAxialCount(fAxialCount);

    KGDiscreteRotationalMeshSurface* tDiscreteRotationalMeshSurface;
    for (auto& surface : fSurfaces) {
        tDiscreteRotationalMeshSurface = surface->MakeExtension<KGDiscreteRotationalMesh>();
        surface->AcceptNode(&tMesher);
        tDiscreteRotationalMeshSurface->SetName(GetName());
        tDiscreteRotationalMeshSurface->SetTags(GetTags());
    }
    KGDiscreteRotationalMeshSpace* tDiscreteRotationalMeshSpace;
    for (auto& space : fSpaces) {
        tDiscreteRotationalMeshSpace = space->MakeExtension<KGDiscreteRotationalMesh>();
        space->AcceptNode(&tMesher);
        tDiscreteRotationalMeshSpace->SetName(GetName());
        tDiscreteRotationalMeshSpace->SetTags(GetTags());
    }
}

void KGDiscreteRotationalMeshAttributor::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}
void KGDiscreteRotationalMeshAttributor::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#include "KGInterfaceBuilder.hh"

namespace katrin
{

template<> KGDiscreteRotationalMeshBuilder::~KComplexElement() = default;

STATICINT sKGDiscreteRotationalMeshStructure = KGDiscreteRotationalMeshBuilder::Attribute<std::string>("name") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<double>("angle") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<int>("count") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<std::string>("surfaces") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<std::string>("spaces");

STATICINT sKGDiscreteRotationalMesh =
    KGInterfaceBuilder::ComplexElement<KGDiscreteRotationalMeshAttributor>("discrete_rotational_mesh");

}  // namespace katrin

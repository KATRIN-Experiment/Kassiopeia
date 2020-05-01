#include "KGDiscreteRotationalMeshBuilder.hh"

#include "KGDiscreteRotationalMesher.hh"

using namespace std;
using namespace KGeoBag;

namespace KGeoBag
{

KGDiscreteRotationalMeshAttributor::KGDiscreteRotationalMeshAttributor() :
    fSurfaces(),
    fSpaces(),
    fAxialAngle(0.),
    fAxialCount(100)
{}

KGDiscreteRotationalMeshAttributor::~KGDiscreteRotationalMeshAttributor()
{
    KGDiscreteRotationalMesher tMesher;
    tMesher.SetAxialAngle(fAxialAngle);
    tMesher.SetAxialCount(fAxialCount);

    KGDiscreteRotationalMeshSurface* tDiscreteRotationalMeshSurface;
    for (auto tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++) {
        tDiscreteRotationalMeshSurface = (*tIt)->MakeExtension<KGDiscreteRotationalMesh>();
        (*tIt)->AcceptNode(&tMesher);
        tDiscreteRotationalMeshSurface->SetName(GetName());
        tDiscreteRotationalMeshSurface->SetTags(GetTags());
    }
    KGDiscreteRotationalMeshSpace* tDiscreteRotationalMeshSpace;
    for (auto tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++) {
        tDiscreteRotationalMeshSpace = (*tIt)->MakeExtension<KGDiscreteRotationalMesh>();
        (*tIt)->AcceptNode(&tMesher);
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

template<> KGDiscreteRotationalMeshBuilder::~KComplexElement() {}

STATICINT sKGDiscreteRotationalMeshStructure = KGDiscreteRotationalMeshBuilder::Attribute<string>("name") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<double>("angle") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<int>("count") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<string>("surfaces") +
                                               KGDiscreteRotationalMeshBuilder::Attribute<string>("spaces");

STATICINT sKGDiscreteRotationalMesh =
    KGInterfaceBuilder::ComplexElement<KGDiscreteRotationalMeshAttributor>("discrete_rotational_mesh");

}  // namespace katrin

#include "KGAxialMeshBuilder.hh"

#include "KGAxialMesher.hh"

using namespace std;
using namespace KGeoBag;

namespace KGeoBag
{

KGAxialMeshAttributor::KGAxialMeshAttributor() : fSurfaces(), fSpaces() {}

KGAxialMeshAttributor::~KGAxialMeshAttributor()
{
    KGAxialMesher tMesher;

    KGAxialMeshSurface* tAxialMeshSurface;
    for (auto tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++) {
        tAxialMeshSurface = (*tIt)->MakeExtension<KGAxialMesh>();
        (*tIt)->AcceptNode(&tMesher);
        tAxialMeshSurface->SetName(GetName());
        tAxialMeshSurface->SetTags(GetTags());
    }
    KGAxialMeshSpace* tAxialMeshSpace;
    for (auto tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++) {
        tAxialMeshSpace = (*tIt)->MakeExtension<KGAxialMesh>();
        (*tIt)->AcceptNode(&tMesher);
        tAxialMeshSpace->SetName(GetName());
        tAxialMeshSpace->SetTags(GetTags());
    }
}

void KGAxialMeshAttributor::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}
void KGAxialMeshAttributor::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#include "KGInterfaceBuilder.hh"

namespace katrin
{

template<> KGAxialMeshBuilder::~KComplexElement() {}

STATICINT sKGAxialMeshStructure = KGAxialMeshBuilder::Attribute<string>("name") +
                                  KGAxialMeshBuilder::Attribute<string>("surfaces") +
                                  KGAxialMeshBuilder::Attribute<string>("spaces");

STATICINT sKGAxialMesh = KGInterfaceBuilder::ComplexElement<KGAxialMeshAttributor>("axial_mesh");

}  // namespace katrin

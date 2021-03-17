#include "KGAxialMeshBuilder.hh"

#include "KGAxialMesher.hh"

using namespace std;
using namespace KGeoBag;

namespace KGeoBag
{

KGAxialMeshAttributor::KGAxialMeshAttributor() = default;

KGAxialMeshAttributor::~KGAxialMeshAttributor()
{
    KGAxialMesher tMesher;

    KGAxialMeshSurface* tAxialMeshSurface;
    for (auto& surface : fSurfaces) {
        tAxialMeshSurface = surface->MakeExtension<KGAxialMesh>();
        surface->AcceptNode(&tMesher);
        tAxialMeshSurface->SetName(GetName());
        tAxialMeshSurface->SetTags(GetTags());
    }
    KGAxialMeshSpace* tAxialMeshSpace;
    for (auto& space : fSpaces) {
        tAxialMeshSpace = space->MakeExtension<KGAxialMesh>();
        space->AcceptNode(&tMesher);
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

template<> KGAxialMeshBuilder::~KComplexElement() = default;

STATICINT sKGAxialMeshStructure = KGAxialMeshBuilder::Attribute<std::string>("name") +
                                  KGAxialMeshBuilder::Attribute<std::string>("surfaces") +
                                  KGAxialMeshBuilder::Attribute<std::string>("spaces");

STATICINT sKGAxialMesh = KGInterfaceBuilder::ComplexElement<KGAxialMeshAttributor>("axial_mesh");

}  // namespace katrin

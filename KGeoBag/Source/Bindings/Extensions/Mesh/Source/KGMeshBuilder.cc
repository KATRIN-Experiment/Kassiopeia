#include "KGMeshBuilder.hh"

#include "KGMesher.hh"

using namespace std;
using namespace KGeoBag;

namespace KGeoBag
{

KGMeshAttributor::KGMeshAttributor() = default;

KGMeshAttributor::~KGMeshAttributor()
{
    KGMesher tMesher;

    coremsg(eNormal) << "Generating mesh for <" << fSurfaces.size() << "> surfaces and <" << fSpaces.size() << "> spaces ..." << eom;

    KGMeshSurface* tMeshSurface;
    for (auto& surface : fSurfaces) {
        tMeshSurface = surface->MakeExtension<KGMesh>();
        surface->AcceptNode(&tMesher);
        tMeshSurface->SetName(GetName());
        tMeshSurface->SetTags(GetTags());
    }
    KGMeshSpace* tMeshSpace;
    for (auto& space : fSpaces) {
        tMeshSpace = space->MakeExtension<KGMesh>();
        space->AcceptNode(&tMesher);
        tMeshSpace->SetName(GetName());
        tMeshSpace->SetTags(GetTags());
    }
}

void KGMeshAttributor::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}
void KGMeshAttributor::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#include "KGInterfaceBuilder.hh"

namespace katrin
{

template<> KGMeshBuilder::~KComplexElement() = default;

STATICINT sKGMeshStructure = KGMeshBuilder::Attribute<std::string>("name") +
                             KGMeshBuilder::Attribute<std::string>("surfaces") +
                             KGMeshBuilder::Attribute<std::string>("spaces");

STATICINT sKGMesh = KGInterfaceBuilder::ComplexElement<KGMeshAttributor>("mesh");

}  // namespace katrin

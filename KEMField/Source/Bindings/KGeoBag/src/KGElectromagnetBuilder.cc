#include "KGElectromagnetBuilder.hh"

using namespace KGeoBag;
using namespace std;

namespace KGeoBag
{

KGElectromagnetAttributor::KGElectromagnetAttributor() : fLineCurrent(0.0), fCurrentTurns(1.0), fDirection(1.0) {}

KGElectromagnetAttributor::~KGElectromagnetAttributor()
{
    KGElectromagnetSurface* tElectromagnetSurface;
    for (auto& surface : fSurfaces) {
        tElectromagnetSurface = surface->MakeExtension<KGElectromagnet>();
        tElectromagnetSurface->SetName(GetName());
        tElectromagnetSurface->SetTags(GetTags());
        tElectromagnetSurface->SetLineCurrent(GetLineCurrent());
        tElectromagnetSurface->SetCurrentTurns(GetCurrentTurns());
    }
    KGElectromagnetSpace* tElectromagnetSpace;
    for (auto& space : fSpaces) {
        tElectromagnetSpace = space->MakeExtension<KGElectromagnet>();
        tElectromagnetSpace->SetName(GetName());
        tElectromagnetSpace->SetTags(GetTags());
        tElectromagnetSpace->SetLineCurrent(GetLineCurrent());
        tElectromagnetSpace->SetCurrentTurns(GetCurrentTurns());
    }
}

void KGElectromagnetAttributor::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}
void KGElectromagnetAttributor::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#include "KGInterfaceBuilder.hh"

namespace katrin
{

template<> inline KGElectromagnetBuilder::~KComplexElement() = default;

STATICINT sKGElectromagnetStructure = KGElectromagnetBuilder::Attribute<std::string>("name") +
                                      KGElectromagnetBuilder::Attribute<double>("current") +
                                      KGElectromagnetBuilder::Attribute<double>("scaling_factor") +
                                      KGElectromagnetBuilder::Attribute<double>("num_turns") +
                                      KGElectromagnetBuilder::Attribute<std::string>("direction") +
                                      KGElectromagnetBuilder::Attribute<std::string>("surfaces") +
                                      KGElectromagnetBuilder::Attribute<std::string>("spaces");


STATICINT sKGElectromagnet = KGInterfaceBuilder::ComplexElement<KGElectromagnetAttributor>("electromagnet");

}  // namespace katrin

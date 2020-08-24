#include "KGElectromagnetBuilder.hh"

using namespace KGeoBag;
using namespace std;

namespace KGeoBag
{

KGElectromagnetAttributor::KGElectromagnetAttributor() :
    fSurfaces(),
    fSpaces(),
    fLineCurrent(0.0),
    fCurrentTurns(1.0),
    fDirection(1.0)
{}

KGElectromagnetAttributor::~KGElectromagnetAttributor()
{
    KGElectromagnetSurface* tElectromagnetSurface;
    for (auto tIt = fSurfaces.begin(); tIt != fSurfaces.end(); tIt++) {
        tElectromagnetSurface = (*tIt)->MakeExtension<KGElectromagnet>();
        tElectromagnetSurface->SetName(GetName());
        tElectromagnetSurface->SetTags(GetTags());
        tElectromagnetSurface->SetLineCurrent(GetLineCurrent());
        tElectromagnetSurface->SetCurrentTurns(GetCurrentTurns());
    }
    KGElectromagnetSpace* tElectromagnetSpace;
    for (auto tIt = fSpaces.begin(); tIt != fSpaces.end(); tIt++) {
        tElectromagnetSpace = (*tIt)->MakeExtension<KGElectromagnet>();
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

template<> inline KGElectromagnetBuilder::~KComplexElement() {}

STATICINT sKGElectromagnetStructure =
    KGElectromagnetBuilder::Attribute<string>("name") + KGElectromagnetBuilder::Attribute<double>("current") +
    KGElectromagnetBuilder::Attribute<double>("scaling_factor") + KGElectromagnetBuilder::Attribute<double>("num_turns") +
    KGElectromagnetBuilder::Attribute<string>("direction") + KGElectromagnetBuilder::Attribute<string>("surfaces") +
    KGElectromagnetBuilder::Attribute<string>("spaces");


STATICINT sKGElectromagnet = KGInterfaceBuilder::ComplexElement<KGElectromagnetAttributor>("electromagnet");

}  // namespace katrin

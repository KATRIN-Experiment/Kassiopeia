#include "KGAppearanceBuilder.hh"

using namespace std;
using namespace KGeoBag;

namespace KGeoBag
{

KGAppearanceAttributor::KGAppearanceAttributor() = default;

KGAppearanceAttributor::~KGAppearanceAttributor()
{
    KGAppearanceSurface* tAppearanceSurface;
    for (auto& surface : fSurfaces) {
        tAppearanceSurface = surface->MakeExtension<KGAppearance>();
        tAppearanceSurface->SetName(GetName());
        tAppearanceSurface->SetTags(GetTags());
        tAppearanceSurface->SetColor(GetColor());
        tAppearanceSurface->SetArc(GetArc());
    }
    KGAppearanceSpace* tAppearanceSpace;
    for (auto& space : fSpaces) {
        tAppearanceSpace = space->MakeExtension<KGAppearance>();
        tAppearanceSpace->SetName(GetName());
        tAppearanceSpace->SetTags(GetTags());
        tAppearanceSpace->SetColor(GetColor());
        tAppearanceSpace->SetArc(GetArc());
    }
}

void KGAppearanceAttributor::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}
void KGAppearanceAttributor::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

}  // namespace KGeoBag

#include "KGInterfaceBuilder.hh"

namespace katrin
{

template<> KGAppearanceBuilder::~KComplexElement() = default;

STATICINT sKGAppearanceStructure =
    KGAppearanceBuilder::Attribute<std::string>("name") + KGAppearanceBuilder::Attribute<KGRGBAColor>("color") +
    KGAppearanceBuilder::Attribute<unsigned int>("arc") + KGAppearanceBuilder::Attribute<std::string>("surfaces") +
    KGAppearanceBuilder::Attribute<std::string>("spaces");

STATICINT sKGAppearance = KGInterfaceBuilder::ComplexElement<KGAppearanceAttributor>("appearance");

}  // namespace katrin

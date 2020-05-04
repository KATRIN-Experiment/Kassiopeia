#ifndef KGAPPEARANCEBUILDER_HH_
#define KGAPPEARANCEBUILDER_HH_

#include "KGAppearance.hh"

namespace KGeoBag
{

class KGAppearanceAttributor : public KTagged, public KGAppearanceData
{
  public:
    KGAppearanceAttributor();
    ~KGAppearanceAttributor() override;

  public:
    void AddSurface(KGSurface* aSurface);
    void AddSpace(KGSpace* aSpace);

  private:
    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;
};

}  // namespace KGeoBag

#include "KComplexElement.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGAppearanceAttributor> KGAppearanceBuilder;

template<> inline bool KGAppearanceBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "name") {
        fObject->SetName(aContainer->AsReference<string>());
        return true;
    }
    if (aContainer->GetName() == "color") {
        fObject->SetColor(aContainer->AsReference<KGRGBAColor>());
        return true;
    }
    if (aContainer->GetName() == "arc") {
        fObject->SetArc(aContainer->AsReference<unsigned int>());
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsReference<string>());
        vector<KGSurface*>::iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            coremsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsReference<string>() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        vector<KGSpace*> tSpaces = KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsReference<string>());
        vector<KGSpace*>::iterator tSpaceIt;
        KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            coremsg(eWarning) << "no spaces found for specifier <" << aContainer->AsReference<string>() << ">" << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            fObject->AddSpace(tSpace);
        }
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

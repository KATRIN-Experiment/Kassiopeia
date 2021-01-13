#ifndef KGELECTROMAGNETBUILDER_HH_
#define KGELECTROMAGNETBUILDER_HH_

#include "KField.h"
#include "KGElectromagnet.hh"

namespace KGeoBag
{

class KGElectromagnetAttributor : public katrin::KTagged, public KGElectromagnetData
{
  public:
    KGElectromagnetAttributor();
    ~KGElectromagnetAttributor() override;

  public:
    void AddSurface(KGSurface* aSurface);
    void AddSpace(KGSpace* aSpace);

  private:
    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;
    K_SET_GET(double, LineCurrent)
    K_SET_GET(double, CurrentTurns)
    K_SET_GET(double, Direction)
};

}  // namespace KGeoBag

#include "KComplexElement.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGElectromagnetAttributor> KGElectromagnetBuilder;

template<> inline bool KGElectromagnetBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace KGeoBag;
    using namespace std;

    if (aContainer->GetName() == "name") {
        fObject->SetName(aContainer->AsString());
        return true;
    }
    if (aContainer->GetName() == "current") {
        fObject->SetLineCurrent(aContainer->AsReference<double>());
        return true;
    }
    if (aContainer->GetName() == "scaling_factor" || aContainer->GetName() == "num_turns") {
        fObject->SetCurrentTurns(aContainer->AsReference<double>());
        return true;
    }
    if (aContainer->GetName() == "direction") {
        string tDirection = aContainer->AsString();
        if (tDirection == "clockwise" || tDirection == "normal") {
            fObject->SetDirection(1.0);
            return true;
        }
        if (tDirection == "counter_clockwise" || tDirection == "reversed") {
            fObject->SetDirection(-1.0);
            return true;
        }
        coremsg(eWarning) << "Dont know the direction <" << tDirection << ">" << eom;
        return false;
    }
    if (aContainer->GetName() == "surfaces") {
        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        vector<KGSurface*>::const_iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            coremsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        vector<KGSpace*> tSpaces = KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        vector<KGSpace*>::const_iterator tSpaceIt;
        KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            coremsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
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

template<> inline bool KGElectromagnetBuilder::End()
{
    fObject->SetLineCurrent(fObject->GetLineCurrent() * fObject->GetDirection());
    fObject->SetCurrentTurns(fObject->GetCurrentTurns());
    return true;
}

}  // namespace katrin

#endif

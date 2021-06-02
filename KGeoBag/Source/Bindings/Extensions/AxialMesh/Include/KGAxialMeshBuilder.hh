#ifndef KGAXIALMESHBUILDER_HH_
#define KGAXIALMESHBUILDER_HH_

#include "KGAxialMesh.hh"
#include "KGBindingsMessage.hh"

namespace KGeoBag
{

class KGAxialMeshAttributor : public katrin::KTagged, public KGAxialMeshData
{
  public:
    KGAxialMeshAttributor();
    ~KGAxialMeshAttributor() override;

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

typedef KComplexElement<KGeoBag::KGAxialMeshAttributor> KGAxialMeshBuilder;

template<> inline bool KGAxialMeshBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "name") {
        fObject->SetName(aContainer->AsString());
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        vector<KGSurface*>::iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            bindmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
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
        vector<KGSpace*>::iterator tSpaceIt;
        KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            bindmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
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

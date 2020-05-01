#ifndef KGDISCRETEROTATIONALMESHBUILDER_HH_
#define KGDISCRETEROTATIONALMESHBUILDER_HH_

#include "KGDiscreteRotationalMesh.hh"

namespace KGeoBag
{

class KGDiscreteRotationalMeshAttributor : public KTagged, public KGDiscreteRotationalMeshData
{
  public:
    KGDiscreteRotationalMeshAttributor();
    ~KGDiscreteRotationalMeshAttributor() override;

  public:
    void AddSurface(KGSurface* aSurface);
    void AddSpace(KGSpace* aSpace);

  private:
    vector<KGSurface*> fSurfaces;
    vector<KGSpace*> fSpaces;

  public:
    void SetAxialAngle(double d)
    {
        fAxialAngle = d;
    }
    void SetAxialCount(int i)
    {
        fAxialCount = i;
    }

  private:
    double fAxialAngle;
    int fAxialCount;
};

}  // namespace KGeoBag

#include "KComplexElement.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGDiscreteRotationalMeshAttributor> KGDiscreteRotationalMeshBuilder;

template<> inline bool KGDiscreteRotationalMeshBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "name") {
        fObject->SetName(aContainer->AsReference<string>());
        return true;
    }
    if (aContainer->GetName() == "angle") {
        aContainer->CopyTo(fObject, &KGDiscreteRotationalMeshAttributor::SetAxialAngle);
        return true;
    }
    if (aContainer->GetName() == "count") {
        aContainer->CopyTo(fObject, &KGDiscreteRotationalMeshAttributor::SetAxialCount);
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

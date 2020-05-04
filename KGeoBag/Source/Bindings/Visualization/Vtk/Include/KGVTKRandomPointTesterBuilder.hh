#ifndef _KGeoBag_KGVTKNormalTesterBuilder_hh_
#define _KGeoBag_KGVTKNormalTesterBuilder_hh_

#include "KComplexElement.hh"
#include "KGVTKRandomPointTester.hh"
#include "KGVisualizationMessage.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGVTKRandomPointTester> KGVTKRandomPointTesterBuilder;

template<> inline bool KGVTKRandomPointTesterBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        if (aContainer->AsReference<string>().size() == 0) {
            return true;
        }

        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsReference<string>());
        vector<KGSurface*>::const_iterator tSurfaceIt;
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
        if (aContainer->AsReference<string>().size() == 0) {
            return true;
        }

        vector<KGSpace*> tSpaces = KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsReference<string>());
        vector<KGSpace*>::const_iterator tSpaceIt;
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

    if (aContainer->GetName() == "sample_color") {
        aContainer->CopyTo(fObject, &KGVTKRandomPointTester::SetSampleColor);
        return true;
    }
    if (aContainer->GetName() == "vertex_size") {
        aContainer->CopyTo(fObject, &KGVTKRandomPointTester::SetVertexSize);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

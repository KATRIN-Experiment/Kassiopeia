#ifndef _KGeoBag_KGVTKDistanceTesterBuilder_hh_
#define _KGeoBag_KGVTKDistanceTesterBuilder_hh_

#include "KComplexElement.hh"
#include "KGVTKDistanceTester.hh"
#include "KGVisualizationMessage.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGVTKDistanceTester> KGVTKDistanceTesterBuilder;

template<> inline bool KGVTKDistanceTesterBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        if (aContainer->AsString().size() == 0) {
            return true;
        }

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
        if (aContainer->AsString().size() == 0) {
            return true;
        }

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

    if (aContainer->GetName() == "sample_disk_origin") {
        aContainer->CopyTo(fObject, &KGVTKDistanceTester::SetSampleDiskOrigin);
        return true;
    }
    if (aContainer->GetName() == "sample_disk_normal") {
        aContainer->CopyTo(fObject, &KGVTKDistanceTester::SetSampleDiskNormal);
        return true;
    }
    if (aContainer->GetName() == "sample_disk_radius") {
        aContainer->CopyTo(fObject, &KGVTKDistanceTester::SetSampleDiskRadius);
        return true;
    }
    if (aContainer->GetName() == "sample_count") {
        aContainer->CopyTo(fObject, &KGVTKDistanceTester::SetSampleCount);
        return true;
    }
    if (aContainer->GetName() == "vertex_size") {
        aContainer->CopyTo(fObject, &KGVTKDistanceTester::SetVertexSize);
        return true;
    }

    return false;
}

}  // namespace katrin

#endif

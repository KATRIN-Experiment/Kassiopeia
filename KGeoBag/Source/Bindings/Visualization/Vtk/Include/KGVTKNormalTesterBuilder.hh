#ifndef _KGeoBag_KGVTKNormalTesterBuilder_hh_
#define _KGeoBag_KGVTKNormalTesterBuilder_hh_

#include "KComplexElement.hh"
#include "KGVTKNormalTester.hh"
#include "KGVisualizationMessage.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGVTKNormalTester> KGVTKNormalTesterBuilder;

template<> inline bool KGVTKNormalTesterBuilder::AddAttribute(KContainer* aContainer)
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
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetSampleDiskOrigin);
        return true;
    }
    if (aContainer->GetName() == "sample_disk_normal") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetSampleDiskNormal);
        return true;
    }
    if (aContainer->GetName() == "sample_disk_radius") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetSampleDiskRadius);
        return true;
    }
    if (aContainer->GetName() == "sample_count") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetSampleCount);
        return true;
    }
    if (aContainer->GetName() == "sample_color") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetSampleColor);
        return true;
    }
    if (aContainer->GetName() == "point_color") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetPointColor);
        return true;
    }
    if (aContainer->GetName() == "normal_color") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetNormalColor);
        return true;
    }
    if (aContainer->GetName() == "normal_length") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetNormalLength);
        return true;
    }
    if (aContainer->GetName() == "vertex_size") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetVertexSize);
        return true;
    }
    if (aContainer->GetName() == "line_size") {
        aContainer->CopyTo(fObject, &KGVTKNormalTester::SetLineSize);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

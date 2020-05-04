#ifndef KGROOTGEOMETRYPAINTERBUILDER_HH_
#define KGROOTGEOMETRYPAINTERBUILDER_HH_

#include "KComplexElement.hh"
#include "KGROOTGeometryPainter.hh"
#include "KGVisualizationMessage.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGROOTGeometryPainter> KGROOTGeometryPainterBuilder;

template<> inline bool KGROOTGeometryPainterBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace KGeoBag;
    using namespace std;

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
    if (aContainer->GetName() == "plane_normal") {
        aContainer->CopyTo(fObject, &KGROOTGeometryPainter::SetPlaneNormal);
        return true;
    }
    if (aContainer->GetName() == "plane_point") {
        aContainer->CopyTo(fObject, &KGROOTGeometryPainter::SetPlanePoint);
        return true;
    }
    if (aContainer->GetName() == "swap_axis") {
        aContainer->CopyTo(fObject, &KGROOTGeometryPainter::SetSwapAxis);
        return true;
    }
    if (aContainer->GetName() == "epsilon") {
        aContainer->CopyTo(fObject, &KGROOTGeometryPainter::SetEpsilon);
        return true;
    }

    return false;
}

}  // namespace katrin

#endif

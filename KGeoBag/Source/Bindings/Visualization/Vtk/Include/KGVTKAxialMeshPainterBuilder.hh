#ifndef _KGeoBag_KGVTKAxialMeshPainterBuilder_hh_
#define _KGeoBag_KGVTKAxialMeshPainterBuilder_hh_

#include "KComplexElement.hh"
#include "KGVTKAxialMeshPainter.hh"
#include "KGVisualizationMessage.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGVTKAxialMeshPainter> KGVTKAxialMeshPainterBuilder;

template<> inline bool KGVTKAxialMeshPainterBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KGVTKAxialMeshPainter::SetFile);
        return true;
    }
    if (aContainer->GetName() == "arc_count") {
        aContainer->CopyTo(fObject, &KGVTKAxialMeshPainter::SetArcCount);
        return true;
    }
    if (aContainer->GetName() == "color_mode") {
        string tMode = aContainer->AsReference<string>();

        if (tMode == "modulo") {
            fObject->SetColorMode(KGVTKAxialMeshPainter::sModulo);
            return true;
        }
        if (tMode == "area") {
            fObject->SetColorMode(KGVTKAxialMeshPainter::sArea);
            return true;
        }
        if (tMode == "aspect") {
            fObject->SetColorMode(KGVTKAxialMeshPainter::sAspect);
            return true;
        }

        coremsg(eWarning) << "unknown option <" << tMode << "> for vtk mesh painter color mode" << eom;
        return false;
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
            tSurface->AcceptNode(fObject);
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
            tSpace->AcceptNode(fObject);
        }
        return true;
    }

    return false;
}

}  // namespace katrin

#endif

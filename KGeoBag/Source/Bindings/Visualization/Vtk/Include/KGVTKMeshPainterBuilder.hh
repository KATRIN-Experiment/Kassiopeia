#ifndef _KGeoBag_KGVTKMeshPainterBuilder_hh_
#define _KGeoBag_KGVTKMeshPainterBuilder_hh_

#include "KComplexElement.hh"
#include "KGVTKMeshPainter.hh"
#include "KGBindingsMessage.hh"

namespace katrin
{

typedef KComplexElement<KGeoBag::KGVTKMeshPainter> KGVTKMeshPainterBuilder;

template<> inline bool KGVTKMeshPainterBuilder::AddAttribute(KContainer* aContainer)
{
    using namespace std;
    using namespace KGeoBag;

    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KGVTKMeshPainter::SetFile);
        return true;
    }
    if (aContainer->GetName() == "arc_count") {
        aContainer->CopyTo(fObject, &KGVTKMeshPainter::SetArcCount);
        return true;
    }
    if (aContainer->GetName() == "color_mode") {
        string tMode = aContainer->AsString();

        if (tMode == "modulo") {
            fObject->SetColorMode(KGVTKMeshPainter::sModulo);
            return true;
        }
        if (tMode == "area") {
            fObject->SetColorMode(KGVTKMeshPainter::sArea);
            return true;
        }
        if (tMode == "aspect") {
            fObject->SetColorMode(KGVTKMeshPainter::sAspect);
            return true;
        }

        bindmsg(eWarning) << "unknown option <" << tMode << "> for vtk mesh painter color mode" << eom;
        return false;
    }
    if (aContainer->GetName() == "surfaces") {
        if (aContainer->AsString().size() == 0) {
            return true;
        }

        vector<KGSurface*> tSurfaces = KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        vector<KGSurface*>::const_iterator tSurfaceIt;
        KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            bindmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            tSurface->AcceptNode(fObject);
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
            bindmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
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

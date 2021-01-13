#ifndef Kassiopeia_KSGeoSurfaceBuilder_h_
#define Kassiopeia_KSGeoSurfaceBuilder_h_

#include "KComplexElement.hh"
#include "KSGeoSurface.h"
#include "KSOperatorsMessage.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGeoSurface> KSGeoSurfaceBuilder;

template<> inline bool KSGeoSurfaceBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KSGeoSurface::SetName);
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        std::vector<KGeoBag::KGSurface*> tSurfaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        std::vector<KGeoBag::KGSurface*>::const_iterator tSurfaceIt;
        KGeoBag::KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            oprmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddContent(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        std::vector<KGeoBag::KGSpace*> tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        std::vector<KGeoBag::KGSpace*>::const_iterator tSpaceIt;
        KGeoBag::KGSpace* tSpace;
        const std::vector<KGeoBag::KGSurface*>* tSurfaces;
        std::vector<KGeoBag::KGSurface*>::const_iterator tSurfaceIt;
        KGeoBag::KGSurface* tSurface;

        if (tSpaces.size() == 0) {
            oprmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            tSurfaces = tSpace->GetBoundaries();
            for (tSurfaceIt = tSurfaces->begin(); tSurfaceIt != tSurfaces->end(); tSurfaceIt++) {
                tSurface = *tSurfaceIt;
                fObject->AddContent(tSurface);
            }
        }
        return true;
    }
    return false;
}

template<> inline bool KSGeoSurfaceBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSCommand>() == true) {
        aContainer->ReleaseTo(fObject, &KSGeoSurface::AddCommand);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

#ifndef Kassiopeia_KSTermMinDistanceBuilder_h_
#define Kassiopeia_KSTermMinDistanceBuilder_h_

#include "KComplexElement.hh"
#include "KSOperatorsMessage.h"
#include "KSTermMinDistance.h"


using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSTermMinDistance> KSTermMinDistanceBuilder;

template<> inline bool KSTermMinDistanceBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        std::vector<KGeoBag::KGSurface*> tSurfaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsReference<std::string>());
        std::vector<KGeoBag::KGSurface*>::const_iterator tSurfaceIt;
        KGeoBag::KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            oprmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsReference<std::string>() << ">"
                             << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        std::vector<KGeoBag::KGSpace*> tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsReference<std::string>());
        std::vector<KGeoBag::KGSpace*>::const_iterator tSpaceIt;
        KGeoBag::KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            oprmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsReference<std::string>() << ">"
                             << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            fObject->AddSpace(tSpace);
        }
        return true;
    }
    if (aContainer->GetName() == "min_distance") {
        aContainer->CopyTo(fObject, &KSTermMinDistance::SetMinDistance);
        return true;
    }
    return false;
}


}  // namespace katrin


#endif

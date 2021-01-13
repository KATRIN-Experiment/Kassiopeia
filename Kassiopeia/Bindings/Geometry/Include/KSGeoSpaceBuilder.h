#ifndef Kassiopeia_KSGeoSpaceBuilder_h_
#define Kassiopeia_KSGeoSpaceBuilder_h_

#include "KComplexElement.hh"
#include "KSGeoSpace.h"
#include "KSOperatorsMessage.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KSGeoSpace> KSGeoSpaceBuilder;

template<> inline bool KSGeoSpaceBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KSGeoSpace::SetName);
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        std::vector<KGeoBag::KGSpace*> tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        std::vector<KGeoBag::KGSpace*>::iterator tSpaceIt;
        KGeoBag::KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            oprmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return false;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            fObject->AddContent(tSpace);
        }
        return true;
    }
    return false;
}

template<> inline bool KSGeoSpaceBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KSGeoSpace>() == true) {
        aContainer->ReleaseTo(fObject, &KSGeoSpace::AddSpace);
        return true;
    }
    if (aContainer->Is<KSGeoSurface>() == true) {
        aContainer->ReleaseTo(fObject, &KSGeoSpace::AddSurface);
        return true;
    }
    if (aContainer->Is<KSGeoSide>() == true) {
        aContainer->ReleaseTo(fObject, &KSGeoSpace::AddSide);
        return true;
    }
    if (aContainer->Is<KSCommand>() == true) {
        aContainer->ReleaseTo(fObject, &KSGeoSpace::AddCommand);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

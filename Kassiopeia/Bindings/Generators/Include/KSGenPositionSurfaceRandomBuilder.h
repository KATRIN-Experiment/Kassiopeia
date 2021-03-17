/*
 * KSGenPositionSurfaceRadnomBuilder.h
 *
 *  Created on: 17.09.2014
 *      Author: J. Behrens
 */

#ifndef KSGENPOSITIONSURFACERANDOMBUILDER_H_
#define KSGENPOSITIONSURFACERANDOMBUILDER_H_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenPositionSurfaceRandom.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
typedef KComplexElement<KSGenPositionSurfaceRandom> KSGenPositionSurfaceRandomBuilder;

template<> inline bool KSGenPositionSurfaceRandomBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }

    if (aContainer->GetName() == "surfaces") {
        std::vector<KGeoBag::KGSurface*> tSurfaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        std::vector<KGeoBag::KGSurface*>::iterator tSurfaceIt;
        KGeoBag::KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            genmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }

    return false;
}
}  // namespace katrin

#endif /* KSGENPOSITIONSURFACERANDOMBUILDER_H_ */

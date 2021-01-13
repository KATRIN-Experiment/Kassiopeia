/*
 * KSGenPositionMeshSurfaceRandom.cxx
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#ifndef _KSGenPositionMeshSurfaceRandom_h_
#define _KSGenPositionMeshSurfaceRandom_h_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenCreator.h"
#include "KSGenPositionMeshSurfaceRandom.h"
#include "KSGenValue.h"
#include "KSGeneratorsMessage.h"
#include "KSRootBuilder.h"

#include <vector>

using namespace Kassiopeia;
namespace katrin
{
typedef KComplexElement<KSGenPositionMeshSurfaceRandom> KSGenPositionMeshSurfaceRandomBuilder;

template<> inline bool KSGenPositionMeshSurfaceRandomBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        if (aContainer->AsString().size() == 0) {
            return true;
        }

        std::vector<KGeoBag::KGSurface*> tSurfaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());
        std::vector<KGeoBag::KGSurface*>::const_iterator tSurfaceIt;
        KGeoBag::KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            genmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            tSurface->AcceptNode(fObject);
        }
        return true;
    }
    return false;
}


}  // namespace katrin

#endif /*_KSGenPositionMeshSurfaceRandom_h_*/

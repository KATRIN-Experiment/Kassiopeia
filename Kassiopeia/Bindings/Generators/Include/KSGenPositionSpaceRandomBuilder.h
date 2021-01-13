/*
 * KSGenPositionSpaceBuilder.h
 *
 *  Created on: 01.07.2014
 *      Author: oertlin
 */

#ifndef KSGENPOSITIONSPACERANDOMBUILDER_H_
#define KSGENPOSITIONSPACERANDOMBUILDER_H_

#include "KComplexElement.hh"
#include "KGCore.hh"
#include "KSGenPositionSpaceRandom.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;

namespace katrin
{
typedef KComplexElement<KSGenPositionSpaceRandom> KSGenPositionSpaceRandomBuilder;

template<> inline bool KSGenPositionSpaceRandomBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }

    if (aContainer->GetName() == "spaces") {
        std::vector<KGeoBag::KGSpace*> tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        std::vector<KGeoBag::KGSpace*>::iterator tSpaceIt;
        KGeoBag::KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            genmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            fObject->AddSpace(tSpace);
        }
        return true;
    }

    return false;
}
}  // namespace katrin

#endif /* KSGENPOSITIONSPACEBUILDER_H_ */

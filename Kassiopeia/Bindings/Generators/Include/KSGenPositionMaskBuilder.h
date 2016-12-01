/*
 * KSGenPositionMaskBuilder.h
 *
 *  Created on: 25.04.2015
 *      Author: J. Behrens
 */

#ifndef _KSGenPositionMaskBuilder_H_
#define _KSGenPositionMaskBuilder_H_

#include "KGCore.hh"
#include "KSGenPositionMask.h"
#include "KComplexElement.hh"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement<KSGenPositionMask> KSGenPositionMaskBuilder;

    template<>
    inline bool KSGenPositionMaskBuilder::AddAttribute(KContainer* aContainer)
    {
        if(aContainer->GetName() == "name")
        {
            aContainer->CopyTo(fObject, &KNamed::SetName);
            return true;
        }
        if(aContainer->GetName() == "spaces_allowed")
        {
            std::vector< KGeoBag::KGSpace* > tSpaces = KGeoBag::KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< std::string >() );
            if(tSpaces.size() == 0)
            {
                genmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsReference<std::string>() << ">" << eom;
                return false;
            }

            for(vector< KGeoBag::KGSpace* >::iterator tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++)
            {
                fObject->AddAllowedSpace((*tSpaceIt));
            }
            return true;
        }
        if(aContainer->GetName() == "spaces_forbidden")
        {
            std::vector< KGeoBag::KGSpace* > tSpaces = KGeoBag::KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< std::string >() );
            if(tSpaces.size() == 0)
            {
                genmsg(eWarning) << "no spaces found for specifier <" << aContainer->AsReference<std::string>() << ">" << eom;
                return false;
            }

            for(vector< KGeoBag::KGSpace* >::iterator tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++)
            {
                fObject->AddForbiddenSpace((*tSpaceIt));
            }
            return true;
        }
        if(aContainer->GetName() == "max_retries")
        {
            aContainer->CopyTo(fObject, &KSGenPositionMask::SetMaxRetries);
            return true;
        }
        return false;
    }

    template<>
    inline bool KSGenPositionMaskBuilder::AddElement(KContainer* aContainer)
    {
        if( aContainer->GetName().substr( 0, 8 ) == "position" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionMask::SetGenerator );
            return true;
        }
        return false;
    }
}

#endif /* _KSGenPositionMaskBuilder_H_ */

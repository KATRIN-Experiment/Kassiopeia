/*
 * KSGenPositionMeshSurfaceRandom.cxx
 *
 *  Created on: 28.01.2015
 *      Author: Nikolaus Trost
 */

#ifndef _KSGenPositionMeshSurfaceRandom_h_
#define _KSGenPositionMeshSurfaceRandom_h_

#include "KComplexElement.hh"
#include "KSRootBuilder.h"
#include "KSGenPositionMeshSurfaceRandom.h"
#include "KGCore.hh"
#include "KSGeneratorsMessage.h"
#include "KSGenCreator.h"
#include "KSGenValue.h"
#include <vector>

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSGenPositionMeshSurfaceRandom > KSGenPositionMeshSurfaceRandomBuilder;

    template< >
    inline bool KSGenPositionMeshSurfaceRandomBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "surfaces" )
        {
            if( aContainer->AsReference< std::string >().size() == 0 )
            {
                return true;
            }

            std::vector< KGeoBag::KGSurface* > tSurfaces = KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< std::string >() );
            std::vector< KGeoBag::KGSurface* >::const_iterator tSurfaceIt;
            KGeoBag::KGSurface* tSurface;

            if( tSurfaces.size() == 0 )
            {
                genmsg( eWarning ) << "no surfaces found for specifier <" << aContainer->AsReference< std::string >() << ">" << eom;
                return true;
            }

            for( tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++ )
            {
                tSurface = *tSurfaceIt;
                tSurface->AcceptNode( fObject );
            }
            return true;
        }
        return false;
    }


}

#endif /*_KSGenPositionMeshSurfaceRandom_h_*/

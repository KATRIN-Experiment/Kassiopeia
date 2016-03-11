#ifndef Kassiopeia_KSGenDirectionSurfaceCompositeBuilder_h_
#define Kassiopeia_KSGenDirectionSurfaceCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenDirectionSurfaceComposite.h"
#include "KSGeneratorsMessage.h"
#include "KSToolbox.h"
#include "KGCore.hh"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

    typedef KComplexElement< KSGenDirectionSurfaceComposite > KSGenDirectionSurfaceCompositeBuilder;

    template< >
    inline bool KSGenDirectionSurfaceCompositeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "outside" )
        {
            aContainer->CopyTo( fObject, &KSGenDirectionSurfaceComposite::SetSide );
            return true;
        }
        if(aContainer->GetName() == "surfaces")
        {
            vector< KGeoBag::KGSurface* > tSurfaces = KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces( aContainer->AsReference< string >() );
            vector< KGeoBag::KGSurface* >::iterator tSurfaceIt;
            KGeoBag::KGSurface* tSurface;

            if(tSurfaces.size() == 0) {
                genmsg(eWarning) << "no surfaces found for specifier <" << aContainer->AsReference<string>() << ">" << eom;
                return false;
            }

            for(tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
                tSurface = *tSurfaceIt;
                fObject->AddSurface(tSurface);
            }
            return true;
        }

        if( aContainer->GetName() == "theta" )
        {
            fObject->SetThetaValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        if( aContainer->GetName() == "phi" )
        {
            fObject->SetPhiValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenDirectionSurfaceCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 5 ) == "theta" )
        {
            aContainer->ReleaseTo( fObject, &KSGenDirectionSurfaceComposite::SetThetaValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 3 ) == "phi" )
        {
            aContainer->ReleaseTo( fObject, &KSGenDirectionSurfaceComposite::SetPhiValue );
            return true;
        }
        return false;
    }

}

#endif

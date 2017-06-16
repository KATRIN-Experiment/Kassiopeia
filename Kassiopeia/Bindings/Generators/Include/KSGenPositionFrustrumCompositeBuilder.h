#ifndef Kassiopeia_KSGenPositionFrustrumCompositeBuilder_h_
#define Kassiopeia_KSGenPositionFrustrumCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenPositionFrustrumComposite.h"
#include "KToolbox.h"
#include "KGCore.hh"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

    typedef KComplexElement< KSGenPositionFrustrumComposite > KSGenPositionFrustrumCompositeBuilder;

    template< >
    inline bool KSGenPositionFrustrumCompositeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "surface" )
        {
            //This surface is never used ?
            //KGSurface* tSurface = KGInterface::GetInstance()->RetrieveSurface( aContainer->AsReference< std::string >() );
            return true;
        }
        if( aContainer->GetName() == "space" )
        {
            //This space is never used ?
            //KGSpace* tSpace = KGInterface::GetInstance()->RetrieveSpace( aContainer->AsReference< std::string >() );
            return true;
        }
        if( aContainer->GetName() == "r" )
        {
            fObject->SetRValue( KToolbox::GetInstance().Get< KSGenValue >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        if( aContainer->GetName() == "phi" )
        {
            fObject->SetPhiValue( KToolbox::GetInstance().Get< KSGenValue >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        if( aContainer->GetName() == "z" )
        {
            fObject->SetZValue( KToolbox::GetInstance().Get< KSGenValue >( aContainer->AsReference< std::string >() ) );
            return true;
        }
        if( aContainer->GetName() == "r1" )
        {
            aContainer->CopyTo( fObject, &KSGenPositionFrustrumComposite::SetR1Value );
            return true;
        }
        if( aContainer->GetName() == "r2" )
        {
            aContainer->CopyTo( fObject, &KSGenPositionFrustrumComposite::SetR2Value );
            return true;
        }
        if( aContainer->GetName() == "z1" )
        {
            aContainer->CopyTo( fObject, &KSGenPositionFrustrumComposite::SetZ1Value );
            return true;
        }
        if( aContainer->GetName() == "z2" )
        {
            aContainer->CopyTo( fObject, &KSGenPositionFrustrumComposite::SetZ2Value );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenPositionFrustrumCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 2 ) == "r1" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionFrustrumComposite::SetR1Value );
            return true;
        }
        if( aContainer->GetName().substr( 0, 2 ) == "r2" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionFrustrumComposite::SetR2Value );
            return true;
        }
        if( aContainer->GetName().substr( 0, 1 ) == "r" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionFrustrumComposite::SetRValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 3 ) == "phi" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionFrustrumComposite::SetPhiValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 2 ) == "z1" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionFrustrumComposite::SetZ1Value );
            return true;
        }
        if( aContainer->GetName().substr( 0, 2 ) == "z2" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionFrustrumComposite::SetZ2Value );
            return true;
        }
        if( aContainer->GetName().substr( 0, 1 ) == "z" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionFrustrumComposite::SetZValue );
            return true;
        }
        return false;
    }

}

#endif

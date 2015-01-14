#ifndef Kassiopeia_KSGenPositionCylindricalCompositeBuilder_h_
#define Kassiopeia_KSGenPositionCylindricalCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenPositionCylindricalComposite.h"
#include "KSToolbox.h"
#include "KGCore.hh"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

    typedef KComplexElement< KSGenPositionCylindricalComposite > KSGenPositionCylindricalCompositeBuilder;

    template< >
    inline bool KSGenPositionCylindricalCompositeBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "surface" )
        {
            KGSurface* tSurface = KGInterface::GetInstance()->RetrieveSurface( aContainer->AsReference< string >() );
            fObject->SetOrigin( tSurface->GetOrigin() );
            fObject->SetXAxis( tSurface->GetXAxis() );
            fObject->SetYAxis( tSurface->GetYAxis() );
            fObject->SetZAxis( tSurface->GetZAxis() );
            return true;
        }
        if( aContainer->GetName() == "space" )
        {
            KGSpace* tSpace = KGInterface::GetInstance()->RetrieveSpace( aContainer->AsReference< string >() );
            fObject->SetOrigin( tSpace->GetOrigin() );
            fObject->SetXAxis( tSpace->GetXAxis() );
            fObject->SetYAxis( tSpace->GetYAxis() );
            fObject->SetZAxis( tSpace->GetZAxis() );
            return true;
        }
        if( aContainer->GetName() == "r" )
        {
            fObject->SetRValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        if( aContainer->GetName() == "phi" )
        {
            fObject->SetPhiValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        if( aContainer->GetName() == "z" )
        {
            fObject->SetZValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        return false;
    }

    template< >
    inline bool KSGenPositionCylindricalCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 1 ) == "r" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionCylindricalComposite::SetRValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 3 ) == "phi" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionCylindricalComposite::SetPhiValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 1 ) == "z" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionCylindricalComposite::SetZValue );
            return true;
        }
        return false;
    }

}

#endif

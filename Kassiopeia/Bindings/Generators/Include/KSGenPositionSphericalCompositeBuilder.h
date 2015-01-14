#ifndef Kassiopeia_KSGenPositionSphericalCompositeBuilder_h_
#define Kassiopeia_KSGenPositionSphericalCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenPositionSphericalComposite.h"
#include "KSToolbox.h"
#include "KGCore.hh"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

    typedef KComplexElement< KSGenPositionSphericalComposite > KSGenPositionSphericalCompositeBuilder;

    template< >
    inline bool KSGenPositionSphericalCompositeBuilder::AddAttribute( KContainer* aContainer )
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
    inline bool KSGenPositionSphericalCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 1 ) == "r" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionSphericalComposite::SetRValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 5 ) == "theta" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionSphericalComposite::SetThetaValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 3 ) == "phi" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionSphericalComposite::SetPhiValue );
            return true;
        }
        return false;
    }

}

#endif

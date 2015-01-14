#ifndef Kassiopeia_KSGenPositionRectangularCompositeBuilder_h_
#define Kassiopeia_KSGenPositionRectangularCompositeBuilder_h_

#include "KComplexElement.hh"
#include "KSGenPositionRectangularComposite.h"
#include "KSToolbox.h"
#include "KGCore.hh"

using namespace Kassiopeia;
using namespace KGeoBag;
namespace katrin
{

    typedef KComplexElement< KSGenPositionRectangularComposite > KSGenPositionRectangularCompositeBuilder;

    template< >
    inline bool KSGenPositionRectangularCompositeBuilder::AddAttribute( KContainer* aContainer )
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
        if( aContainer->GetName() == "x" )
        {
            fObject->SetXValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
            return true;
        }
        if( aContainer->GetName() == "y" )
        {
            fObject->SetYValue( KSToolbox::GetInstance()->GetObjectAs< KSGenValue >( aContainer->AsReference< string >() ) );
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
    inline bool KSGenPositionRectangularCompositeBuilder::AddElement( KContainer* aContainer )
    {
        if( aContainer->GetName().substr( 0, 1 ) == "x" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionRectangularComposite::SetXValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 1 ) == "y" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionRectangularComposite::SetYValue );
            return true;
        }
        if( aContainer->GetName().substr( 0, 1 ) == "z" )
        {
            aContainer->ReleaseTo( fObject, &KSGenPositionRectangularComposite::SetZValue );
            return true;
        }
        return false;
    }

}

#endif

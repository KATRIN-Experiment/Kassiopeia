#ifndef Kassiopeia_KSFieldElectricPotentialmapBuilder_h_
#define Kassiopeia_KSFieldElectricPotentialmapBuilder_h_

#include "KSFieldElectricPotentialmap.h"
#include "KSFieldElectricConstantBuilder.h"
#include "KSFieldElectricQuadrupoleBuilder.h"
#include "KSFieldElectrostaticBuilder.h"
#include "KComplexElement.hh"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldElectricPotentialmap > KSFieldElectricPotentialmapBuilder;

    template< >
    inline bool KSFieldElectricPotentialmapBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmap::SetName );
            return true;
        }
        if( aContainer->GetName() == "directory" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmap::SetDirectory );
            return true;
        }
        if( aContainer->GetName() == "file" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmap::SetFile );
            return true;
        }
        if( aContainer->GetName() == "interpolation" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmap::SetInterpolation );
            return true;
        }
        return false;
    }

    ////////////////////////////////////////////////////////////////////

    typedef KComplexElement< KSFieldElectricPotentialmapCalculator > KSFieldElectricPotentialmapCalculatorBuilder;

    template< >
    inline bool KSFieldElectricPotentialmapCalculatorBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "directory" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetDirectory );
            return true;
        }
        if( aContainer->GetName() == "file" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetFile );
            return true;
        }
        if( aContainer->GetName() == "center" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetCenter );
            return true;
        }
        if( aContainer->GetName() == "length" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetLength );
            return true;
        }
        if( aContainer->GetName() == "mirror_x" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetMirrorX );
            return true;
        }
        if( aContainer->GetName() == "mirror_y" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetMirrorY );
            return true;
        }
        if( aContainer->GetName() == "mirror_z" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetMirrorZ );
            return true;
        }
        if( aContainer->GetName() == "spacing" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricPotentialmapCalculator::SetSpacing );
            return true;
        }
        if( aContainer->GetName() == "spaces" )
        {
            vector< KGSpace* > tSpaces = KGInterface::GetInstance()->RetrieveSpaces( aContainer->AsReference< string >() );
            vector< KGSpace* >::const_iterator tSpaceIt;
            KGSpace* tSpace;

            if( tSpaces.size() == 0 )
            {
                coremsg( eWarning ) << "no spaces found for specifier <" << aContainer->AsReference< string >() << ">" << eom;
                return false;
            }

            for( tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++ )
            {
                tSpace = *tSpaceIt;
                fObject->AddSpace( tSpace );
            }
            return true;
        }
        return false;
    }

    template< >
    inline bool KSFieldElectricPotentialmapCalculatorBuilder::AddElement( KContainer* anElement )
    {
        if( anElement->GetName().substr( 0, 5 ) == "field" )
        {
            anElement->ReleaseTo( fObject, &KSFieldElectricPotentialmapCalculator::SetElectricField );
            fObject->Initialize();  // explicitely initialize here to calculate potential map directly
            return true;
        }
        return false;
    }

}

#endif

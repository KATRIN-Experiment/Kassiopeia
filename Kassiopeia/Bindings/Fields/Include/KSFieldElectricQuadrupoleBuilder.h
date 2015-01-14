#ifndef Kassiopeia_KSFieldElectricQuadrupoleBuilder_h_
#define Kassiopeia_KSFieldElectricQuadrupoleBuilder_h_

#include "KSFieldElectricQuadrupole.h"
#include "KComplexElement.hh"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSFieldElectricQuadrupole > KSFieldElectricQuadrupoleBuilder;

    template< >
    inline bool KSFieldElectricQuadrupoleBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KSElectricField::SetName );
            return true;
        }
        if( aContainer->GetName() == "location" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricQuadrupole::SetLocation );
            return true;
        }
        if( aContainer->GetName() == "strength" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricQuadrupole::SetStrength );
            return true;
        }
        if( aContainer->GetName() == "length" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricQuadrupole::SetLength );
            return true;
        }
        if( aContainer->GetName() == "radius" )
        {
            aContainer->CopyTo( fObject, &KSFieldElectricQuadrupole::SetRadius );
            return true;
        }
        return false;
    }

}
#endif

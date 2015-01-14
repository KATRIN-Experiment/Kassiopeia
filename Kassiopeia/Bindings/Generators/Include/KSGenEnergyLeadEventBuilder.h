#ifndef KSGENENERGYLEADEVENTBUILDER_H
#define KSGENENERGYLEADEVENTBUILDER_H

#include "KComplexElement.hh"
#include "KSGenEnergyLeadEvent.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KSGenEnergyLeadEvent > KSGenEnergyLeadEventBuilder;

    template< >
    inline bool KSGenEnergyLeadEventBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "force_conversion" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyLeadEvent::SetForceConversion );
            return true;
        }
        if( aContainer->GetName() == "do_conversion" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyLeadEvent::SetDoConversion );
            return true;
        }
        if( aContainer->GetName() == "do_auger" )
        {
            aContainer->CopyTo( fObject, &KSGenEnergyLeadEvent::SetDoAuger );
            return true;
        }
        return false;
    }

}

#endif // KSGENENERGYLEADEVENTBUILDER_H

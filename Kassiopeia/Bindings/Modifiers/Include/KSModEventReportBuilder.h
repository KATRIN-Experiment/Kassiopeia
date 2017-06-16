#ifndef Kassiopeia_KSModEventReportBuilder_h_
#define Kassiopeia_KSModEventReportBuilder_h_

#include "KComplexElement.hh"
#include "KSModEventReport.h"
#include "KToolbox.h"


using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSModEventReport > KSModEventReportBuilder;

    template< >
    inline bool KSModEventReportBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        return false;
    }
}


#endif

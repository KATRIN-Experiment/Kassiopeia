#ifndef KESSELASTICELSEPABUILDER_H
#define KESSELASTICELSEPABUILDER_H

#include "KComplexElement.hh"
#include "KSToolbox.h"
#include "KSInteractionsMessage.h"
#include "KESSElasticElsepa.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KESSElasticElsepa > KESSElasticElsepaBuilder;

    template< >
    inline bool KESSElasticElsepaBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            intmsg_debug( "KESSElasticElsepaBuilder::AddAttribute - KESS uses ElasticElsepa for elastic Scattering!" << eom )
            return true;
        }
        return false;
    }

}
#endif // KESSELASTICELSEPABUILDER_H

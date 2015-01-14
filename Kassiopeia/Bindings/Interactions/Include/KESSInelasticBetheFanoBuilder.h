#ifndef KESSINELASTICBETHEFANOBUILDER_H
#define KESSINELASTICBETHEFANOBUILDER_H
#include "KComplexElement.hh"
#include "KESSInelasticBetheFano.h"
#include "KESSPhotoAbsorbtion.h"
#include "KESSRelaxation.h"
#include "KSInteractionsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

    typedef KComplexElement< KESSInelasticBetheFano > KESSInelasticBetheFanoBuilder;

    template< >
    inline bool KESSInelasticBetheFanoBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "PhotoAbsorption" )
        {
            if ( aContainer->AsReference< bool >()){
                KESSPhotoAbsorbtion* aPhotoabsorption = new KESSPhotoAbsorbtion();
                fObject->SetIonisationCalculator( aPhotoabsorption );
                intmsg_debug( "KESSInelasticBetheFanoBuilder::AddAttribute: added a PhotoAbsorption calculator to KESSInelasticBetheFano" << eom );
            }
            else
            {
                intmsg_debug( "KESSInelasticBetheFanoBuilder::AddAttribute: PhotoAbsorption calculator is not added to KESSInelasticBetheFano" << eom );
            }
            return true;
        }
        if( aContainer->GetName() == "AugerRelaxation" )
        {
            if ( aContainer->AsReference< bool >()){
                KESSRelaxation* aRelaxation = new KESSRelaxation;
                fObject->SetRelaxationCalculator( aRelaxation );
                intmsg_debug( "KESSInelasticBetheFanoBuilder::AddAttribute: added an AugerRelaxation calculator to KESSInelasticBetheFano" << eom );
            }
            else
            {
                intmsg_debug( "KESSInelasticBetheFanoBuilder::AddAttribute: AugerRelaxation calculator is not added to KESSInelasticBetheFano" << eom );
            }
            return true;
        }

        return false;
    }

}

#endif // KESSINELASTICBETHEFANOBUILDER_H

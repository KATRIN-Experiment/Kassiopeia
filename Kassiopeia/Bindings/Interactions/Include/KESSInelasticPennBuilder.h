#ifndef KESSINELASTICPENNBUILDER_H
#define KESSINELASTICPENNBUILDER_H

#include "KComplexElement.hh"
#include "KESSInelasticPenn.h"
#include "KESSPhotoAbsorbtion.h"
#include "KESSRelaxation.h"
#include "KSInteractionsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KESSInelasticPenn> KESSInelasticPennBuilder;

template<> inline bool KESSInelasticPennBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KNamed::SetName);
        return true;
    }
    if (aContainer->GetName() == "PhotoAbsorption") {
        if (aContainer->AsReference<bool>()) {
            auto* aPhotoabsorption = new KESSPhotoAbsorbtion();
            fObject->SetIonisationCalculator(aPhotoabsorption);
            intmsg_debug(
                "KESSInelasticPennBuilder::AddAttribute: added a PhotoAbsorption calculator to KESSInelasticPenn"
                << eom);
        }
        else {
            intmsg_debug(
                "KESSInelasticPennBuilder::AddAttribute: PhotoAbsorption calculator is not added to KESSInelasticPenn"
                << eom);
        }
        return true;
    }
    if (aContainer->GetName() == "AugerRelaxation") {
        if (aContainer->AsReference<bool>()) {
            auto* aRelaxation = new KESSRelaxation;
            fObject->SetRelaxationCalculator(aRelaxation);
            intmsg_debug(
                "KESSInelasticPennBuilder::AddAttribute: added an AugerRelaxation calculator to KESSInelasticPenn"
                << eom);
        }
        else {
            intmsg_debug(
                "KESSInelasticPennBuilder::AddAttribute: AugerRelaxation calculator is not added to KESSInelasticPenn"
                << eom);
        }
        return true;
    }

    return false;
}

}  // namespace katrin

#endif  // KESSINELASTICPENNBUILDER_H

#ifndef KESSSURFACEINTERACTIONBUILDER_H
#define KESSSURFACEINTERACTIONBUILDER_H

#include "KComplexElement.hh"
#include "KESSSurfaceInteraction.h"
#include "KSInteractionsMessage.h"

using namespace Kassiopeia;
namespace katrin
{

typedef KComplexElement<KESSSurfaceInteraction> KESSSurfaceInteractionBuilder;

template<> inline bool KESSSurfaceInteractionBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KESSSurfaceInteraction::SetName);
        return true;
    }

    if (aContainer->GetName() == "siliconside") {
        if (aContainer->AsReference<std::string>() == "inside") {
            fObject->SetSurfaceOrientation(KESSSurfaceInteraction::eNormalPointingAway);
        }
        else if (aContainer->AsReference<std::string>() == "outside") {
            fObject->SetSurfaceOrientation(KESSSurfaceInteraction::eNormalPointingSilicon);
        }
        else {
            intmsg(eError)
                << "KESSSurfaceInteractionBuilder::AddAttribute: For KESSSurfaceInteraction the only siliconside options available are 'inside' and 'outside'. Check your Config File"
                << eom;
        }
        return true;
    }

    return false;
}

}  // namespace katrin

#endif  // KESSSURFACEINTERACTIONBUILDER_H

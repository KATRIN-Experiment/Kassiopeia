#ifndef KAPPLICATIONRUNNERBUILDER_H_
#define KAPPLICATIONRUNNERBUILDER_H_

#include "KComplexElement.hh"
#include "KApplicationRunner.h"
#include <KNamedBuilder.h>
#include "KToolbox.h"

namespace katrin{

typedef KComplexElement<KApplicationRunner> KApplicationRunnerBuilder;

template <>
inline bool KApplicationRunnerBuilder::AddAttribute(KContainer *aToken)
{
    if( aToken->GetName() == "name" )
    {
        aToken->CopyTo( fObject, &KNamed::SetName );
        return true;
    }
    return false;
}

template <>
inline bool KApplicationRunnerBuilder::AddElement(KContainer *anElement)
{
    if( anElement->Is<KApplication>() )
    {
        anElement->ReleaseTo( fObject, &KApplicationRunner::AddApplication);
        return true;
    }
    if( anElement->Is<KNamedReference>() )
    {
        KNamedReference& tRef = anElement->AsReference<KNamedReference>();
        KApplication* app = KToolbox::GetInstance().Get<KApplication>(tRef.GetName());
        if(!app)
            return false;
        fObject->AddApplication(app);
        return true;
    }
    return false;
}

}

#endif
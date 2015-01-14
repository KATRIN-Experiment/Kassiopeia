#ifndef Kassiopeia_KSModDynamicEnhancementBuilder_h_
#define Kassiopeia_KSModDynamicEnhancementBuilder_h_

#include "KComplexElement.hh"
#include "KSModDynamicEnhancement.h"
#include "KSToolbox.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSModDynamicEnhancement > KSModDynamicEnhancementBuilder;

    template< >
    inline bool KSModDynamicEnhancementBuilder::AddAttribute(KContainer *aContainer)
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "synchrotron" )
        {
            fObject->SetSynchrotron( KSToolbox::GetInstance()->GetObjectAs< KSTrajTermSynchrotron >( aContainer->AsReference< string >() ) );
            return true;
        }
        if( aContainer->GetName() == "scattering" )
        {
            fObject->SetScattering( KSToolbox::GetInstance()->GetObjectAs< KSIntScattering >( aContainer->AsReference< string >() ) );
            return true;
        }
        if( aContainer->GetName() == "static_enhancement" )
        {
            aContainer->CopyTo( fObject, &KSModDynamicEnhancement::SetStaticEnhancement );
            return true;
        }
        if( aContainer->GetName() == "dynamic" )
        {
            aContainer->CopyTo( fObject, &KSModDynamicEnhancement::SetDynamic );
            return true;
        }
        if( aContainer->GetName() == "reference_energy" )
        {
            aContainer->CopyTo( fObject, &KSModDynamicEnhancement::SetReferenceCrossSectionEnergy );
            return true;
        }
        return false;
    }
}

#endif // Kassiopeia_KSModDynamicEnhancementBuilder_h_

#include "KSModDynamicEnhancementBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{
    template< >
    KSModDynamicEnhancementBuilder::~KComplexElement()
    {
    }

    static int SKSModDynamicEnhancementStructure =
            KSModDynamicEnhancementBuilder::Attribute< string >( "name" )+
            KSModDynamicEnhancementBuilder::Attribute< string >( "synchrotron" )+
            KSModDynamicEnhancementBuilder::Attribute< string >( "scattering" )+
            KSModDynamicEnhancementBuilder::Attribute< double >( "static_enhancement")+
            KSModDynamicEnhancementBuilder::Attribute< bool >( "dynamic" )+
            KSModDynamicEnhancementBuilder::Attribute< double>( "reference_energy" );

    static int sKSModDynamicEnhancement =
            KSRootBuilder::ComplexElement< KSModDynamicEnhancement >( "ksmod_dynamic_enhancement" );
}
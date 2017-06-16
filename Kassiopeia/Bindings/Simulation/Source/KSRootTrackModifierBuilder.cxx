#include "KSRootTrackModifierBuilder.h"
#include "KSRootBuilder.h"
#include <string>

using namespace Kassiopeia;
namespace katrin
{
    template< >
    KSRootTrackModifierBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootTrackModifier =
            KSRootBuilder::ComplexElement< KSRootTrackModifier >( "ks_root_track_modifier" );

    STATICINT sKSRootTrackModifierStructure =
            KSRootTrackModifierBuilder::Attribute< std::string >( "name" ) +
            KSRootTrackModifierBuilder::Attribute< std::string >( "add_modifier" );
}

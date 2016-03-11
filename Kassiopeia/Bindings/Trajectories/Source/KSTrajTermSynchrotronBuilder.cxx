#include "KSTrajTermSynchrotronBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermSynchrotronBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTermSynchrotronStructure =
        KSTrajTermSynchrotronBuilder::Attribute< string >( "name" ) +
        KSTrajTermSynchrotronBuilder::Attribute< double >( "enhancement" );

    STATICINT sToolboxKSTrajTermSynchrotron =
        KSRootBuilder::ComplexElement< KSTrajTermSynchrotron >( "kstraj_term_synchrotron" );

}

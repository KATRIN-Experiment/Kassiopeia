#include "KSTrajTermSynchrotronBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermSynchrotronBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTermSynchrotronStructure =
        KSTrajTermSynchrotronBuilder::Attribute< string >( "name" ) +
        KSTrajTermSynchrotronBuilder::Attribute< double >( "enhancement" );

    static int sToolboxKSTrajTermSynchrotron =
        KSRootBuilder::ComplexElement< KSTrajTermSynchrotron >( "kstraj_term_synchrotron" );

}

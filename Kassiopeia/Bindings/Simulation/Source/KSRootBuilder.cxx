#include "KSRootBuilder.h"
#include "KElementProcessor.hh"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootBuilder::~KComplexElement()
    {
    }

    static int sKSRoot =
        KElementProcessor::ComplexElement< KSRoot >( "kassiopeia" );

}

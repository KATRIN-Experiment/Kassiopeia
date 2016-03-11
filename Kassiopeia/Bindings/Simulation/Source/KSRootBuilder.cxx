#include "KSRootBuilder.h"
#include "KElementProcessor.hh"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRoot =
        KElementProcessor::ComplexElement< KSRoot >( "kassiopeia" );

}

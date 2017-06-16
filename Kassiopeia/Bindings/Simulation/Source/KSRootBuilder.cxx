#include "KSRootBuilder.h"
#include "KElementProcessor.hh"
#include "KRoot.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSRootBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRoot =
        KRootBuilder::ComplexElement< KSRoot >( "kassiopeia" );

    STATICINT sKSRootCompat =
        KElementProcessor::ComplexElement< KSRoot >( "kassiopeia" );
}

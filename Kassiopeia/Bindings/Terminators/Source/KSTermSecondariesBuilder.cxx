#include "KSTermSecondariesBuilder.h"
#include "KSRootBuilder.h"


using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTermSecondariesBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermSecondariesStructure =
		KSTermSecondariesBuilder::Attribute< string >( "name" );

    STATICINT sToolboxKSTermSecondaries =
		KSRootBuilder::ComplexElement< KSTermSecondaries >( "ksterm_secondaries" );

}

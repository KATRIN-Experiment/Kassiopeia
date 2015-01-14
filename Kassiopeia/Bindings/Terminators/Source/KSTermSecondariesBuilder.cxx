#include "KSTermSecondariesBuilder.h"
#include "KSRootBuilder.h"


using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermSecondariesBuilder::~KComplexElement()
    {
    }

    static int sKSTermSecondariesStructure =
		KSTermSecondariesBuilder::Attribute< string >( "name" );

    static int sToolboxKSTermSecondaries =
		KSRootBuilder::ComplexElement< KSTermSecondaries >( "ksterm_secondaries" );

}

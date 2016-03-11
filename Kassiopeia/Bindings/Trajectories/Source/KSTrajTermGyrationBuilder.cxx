#include "KSTrajTermGyrationBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermGyrationBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTermGyrationStructure =
        KSTrajTermGyrationBuilder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajTermGyration =
        KSRootBuilder::ComplexElement< KSTrajTermGyration >( "kstraj_term_gyration" );

}

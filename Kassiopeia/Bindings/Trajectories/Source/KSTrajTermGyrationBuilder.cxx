#include "KSTrajTermGyrationBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermGyrationBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTermGyrationStructure =
        KSTrajTermGyrationBuilder::Attribute< string >( "name" );

    static int sToolboxKSTrajTermGyration =
        KSRootBuilder::ComplexElement< KSTrajTermGyration >( "kstraj_term_gyration" );

}

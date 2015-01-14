#include "KSTrajTermDriftBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermDriftBuilder::~KComplexElement()
    {
    }

    static int sKSTrajTermDriftStructure =
        KSTrajTermDriftBuilder::Attribute< string >( "name" );

    static int sToolboxKSTrajTermDrift =
        KSRootBuilder::ComplexElement< KSTrajTermDrift >( "kstraj_term_drift" );

}

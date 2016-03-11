#include "KSTrajTermDriftBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTrajTermDriftBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTrajTermDriftStructure =
        KSTrajTermDriftBuilder::Attribute< string >( "name" );

    STATICINT sToolboxKSTrajTermDrift =
        KSRootBuilder::ComplexElement< KSTrajTermDrift >( "kstraj_term_drift" );

}

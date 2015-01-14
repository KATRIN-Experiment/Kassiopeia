#include "KESSElasticElsepaBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KESSElasticElsepaBuilder::~KComplexElement()
    {
    }

    static int sKSElasticElsepaStructure = KESSElasticElsepaBuilder::Attribute< string >( "name" );

    static int sKSElasticElsepa = KSRootBuilder::ComplexElement< KESSElasticElsepa >( "kess_elastic_elsepa" );

}

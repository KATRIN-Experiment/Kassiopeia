#include "KESSElasticElsepaBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KESSElasticElsepaBuilder::~KComplexElement()
    {
    }

    STATICINT sKSElasticElsepaStructure = KESSElasticElsepaBuilder::Attribute< string >( "name" );

    STATICINT sKSElasticElsepa = KSRootBuilder::ComplexElement< KESSElasticElsepa >( "kess_elastic_elsepa" );

}

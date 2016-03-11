#include "KESSInelasticPennBuilder.h"
#include "KSRootBuilder.h"


using namespace Kassiopeia;
namespace katrin
{

    template< >
    KESSInelasticPennBuilder::~KComplexElement()
    {
    }

    STATICINT sKSInelasticPennStructure = KESSInelasticPennBuilder::Attribute< string >( "name" )
                                         + KESSInelasticPennBuilder::Attribute< bool >( "PhotoAbsorption" )
                                         + KESSInelasticPennBuilder::Attribute< bool >( "AugerRelaxation" );

    STATICINT sKSInelasticPenn = KSRootBuilder::ComplexElement< KESSInelasticPenn >( "kess_inelastic_penn" );

}

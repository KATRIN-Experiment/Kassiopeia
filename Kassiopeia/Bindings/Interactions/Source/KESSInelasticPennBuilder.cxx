#include "KESSInelasticPennBuilder.h"
#include "KSRootBuilder.h"


using namespace Kassiopeia;
namespace katrin
{

    template< >
    KESSInelasticPennBuilder::~KComplexElement()
    {
    }

    static int sKSInelasticPennStructure = KESSInelasticPennBuilder::Attribute< string >( "name" )
                                         + KESSInelasticPennBuilder::Attribute< bool >( "PhotoAbsorption" )
                                         + KESSInelasticPennBuilder::Attribute< bool >( "AugerRelaxation" );

    static int sKSInelasticPenn = KSRootBuilder::ComplexElement< KESSInelasticPenn >( "kess_inelastic_penn" );

}

#include "KSGenValueRadiusSpherical.h"

#include "KSGeneratorsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSGenValueRadiusSpherical::KSGenValueRadiusSpherical() :
            fRadiusMin( 0. ),
            fRadiusMax( 0. )
    {
    }
    KSGenValueRadiusSpherical::KSGenValueRadiusSpherical( const KSGenValueRadiusSpherical& aCopy ) :
            KSComponent(),
            fRadiusMin( aCopy.fRadiusMin ),
            fRadiusMax( aCopy.fRadiusMax )
    {
    }
    KSGenValueRadiusSpherical* KSGenValueRadiusSpherical::Clone() const
    {
        return new KSGenValueRadiusSpherical( *this );
    }
    KSGenValueRadiusSpherical::~KSGenValueRadiusSpherical()
    {
    }

    void KSGenValueRadiusSpherical::DiceValue( vector< double >& aDicedValues )
    {
        double tMinRadiusCubed = fRadiusMin * fRadiusMin * fRadiusMin;
        double tMaxRadiusCubed = fRadiusMax * fRadiusMax * fRadiusMax;
        double tRadius = pow( KRandom::GetInstance().Uniform( tMinRadiusCubed, tMaxRadiusCubed ), (1./3.) );

        aDicedValues.push_back( tRadius );

        return;
    }

}

#include "KSGenValueRadiusCylindrical.h"

#include "KSGeneratorsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSGenValueRadiusCylindrical::KSGenValueRadiusCylindrical() :
            fRadiusMin( 0. ),
            fRadiusMax( 0. )
    {
    }
    KSGenValueRadiusCylindrical::KSGenValueRadiusCylindrical( const KSGenValueRadiusCylindrical& aCopy ) :
            fRadiusMin( aCopy.fRadiusMin ),
            fRadiusMax( aCopy.fRadiusMax )
    {
    }
    KSGenValueRadiusCylindrical* KSGenValueRadiusCylindrical::Clone() const
    {
        return new KSGenValueRadiusCylindrical( *this );
    }
    KSGenValueRadiusCylindrical::~KSGenValueRadiusCylindrical()
    {
    }

    void KSGenValueRadiusCylindrical::DiceValue( vector< double >& aDicedValues )
    {
        double tMinRadiusSquared = fRadiusMin * fRadiusMin;
        double tMaxRadiusSquared = fRadiusMax * fRadiusMax;
        double tRadius = pow( KRandom::GetInstance().Uniform( tMinRadiusSquared, tMaxRadiusSquared ), (1./2.) );

        aDicedValues.push_back( tRadius );

        return;
    }

}

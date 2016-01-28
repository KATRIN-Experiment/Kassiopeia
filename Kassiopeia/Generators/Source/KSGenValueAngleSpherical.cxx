#include "KSGenValueAngleSpherical.h"

#include "KSGeneratorsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSGenValueAngleSpherical::KSGenValueAngleSpherical() :
            fAngleMin( 0. ),
            fAngleMax( 0. )
    {
    }
    KSGenValueAngleSpherical::KSGenValueAngleSpherical( const KSGenValueAngleSpherical& aCopy ) :
            KSComponent(),
            fAngleMin( aCopy.fAngleMin ),
            fAngleMax( aCopy.fAngleMax )
    {
    }
    KSGenValueAngleSpherical* KSGenValueAngleSpherical::Clone() const
    {
        return new KSGenValueAngleSpherical( *this );
    }
    KSGenValueAngleSpherical::~KSGenValueAngleSpherical()
    {
    }

    void KSGenValueAngleSpherical::DiceValue( vector< double >& aDicedValues )
    {
        double tCosThetaMin = cos( (KConst::Pi() / 180.) * fAngleMax );
        double tCosThetaMax = cos( (KConst::Pi() / 180.) * fAngleMin );
        double tAngle = acos( KRandom::GetInstance().Uniform( tCosThetaMin, tCosThetaMax ) );

        aDicedValues.push_back( (180.0 / KConst::Pi()) * tAngle );

        return;
    }

}

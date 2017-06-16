#include "KSGenValueAngleCosine.h"

#include "KSGeneratorsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSGenValueAngleCosine::KSGenValueAngleCosine() :
            fAngleMin( 0. ),
            fAngleMax( 0. )
    {
    }
    KSGenValueAngleCosine::KSGenValueAngleCosine( const KSGenValueAngleCosine& aCopy ) :
            KSComponent(),
            fAngleMin( aCopy.fAngleMin ),
            fAngleMax( aCopy.fAngleMax )
    {
    }
    KSGenValueAngleCosine* KSGenValueAngleCosine::Clone() const
    {
        return new KSGenValueAngleCosine( *this );
    }
    KSGenValueAngleCosine::~KSGenValueAngleCosine()
    {
    }

    void KSGenValueAngleCosine::DiceValue( vector< double >& aDicedValues )
    {
        double tSinThetaMin = sin( (KConst::Pi() / 180.) * fAngleMin );
        double tSinThetaMax = sin( (KConst::Pi() / 180.) * fAngleMax );

        double tSinTheta = KRandom::GetInstance().Uniform( tSinThetaMin, tSinThetaMax );
        double tAngle = acos( sqrt( 1. - tSinTheta*tSinTheta ) );

        aDicedValues.push_back( (180.0 / KConst::Pi()) * tAngle );

        return;
    }

}

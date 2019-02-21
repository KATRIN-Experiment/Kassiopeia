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
        
        double tsinThetaSquaredMin = pow(sin( (KConst::Pi() / 180.) * fAngleMin ),2);
        double tsinThetaSquaredMax = pow(sin( (KConst::Pi() / 180.) * fAngleMax ),2);
        
        //Random generation follows Eq. 12 from J. Greenwood, Vacuum, 67 (2002), pp. 217-222
        double tsinThetaSquared = KRandom::GetInstance().Uniform( tsinThetaSquaredMin, tsinThetaSquaredMax );
        double tAngle = asin( sqrt(tsinThetaSquared) );
        
        aDicedValues.push_back( (180.0 / KConst::Pi()) * tAngle );

        return;
    }

}

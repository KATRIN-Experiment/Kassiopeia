#include "KSGenValueRadiusFraction.h"

#include "KSGeneratorsMessage.h"

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"
using katrin::KConst;

namespace Kassiopeia
{

    KSGenValueRadiusFraction::KSGenValueRadiusFraction( )
    {
    }

    KSGenValueRadiusFraction::KSGenValueRadiusFraction( const KSGenValueRadiusFraction& /*aCopy*/ ) :
            KSComponent()
    {
    }
    KSGenValueRadiusFraction* KSGenValueRadiusFraction::Clone() const
    {
        return new KSGenValueRadiusFraction( *this );
    }
    KSGenValueRadiusFraction::~KSGenValueRadiusFraction()
    {
    }

    void KSGenValueRadiusFraction::DiceValue( vector< double >& aDicedValues )
    {

        double tRadiusF = pow( KRandom::GetInstance().Uniform( 0., 1. ), (1./2.) );

        aDicedValues.push_back( tRadiusF );

        return;
    }

}

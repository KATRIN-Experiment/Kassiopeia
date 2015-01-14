#include "KSGenValueFix.h"

namespace Kassiopeia
{

    KSGenValueFix::KSGenValueFix() :
            fValue( 0. )
    {
    }
    KSGenValueFix::KSGenValueFix( const KSGenValueFix& aCopy ) :
            fValue( aCopy.fValue )
    {
    }
    KSGenValueFix* KSGenValueFix::Clone() const
    {
        return new KSGenValueFix( *this );
    }
    KSGenValueFix::~KSGenValueFix()
    {
    }

    void KSGenValueFix::DiceValue( vector< double >& aDicedValues )
    {
        double tValue = fValue;
        aDicedValues.push_back( tValue );
    }

}

#include "KSGenValueSet.h"

namespace Kassiopeia
{

    KSGenValueSet::KSGenValueSet() :
            fValueStart( 0. ),
            fValueStop( 0. ),
            fValueCount( 0 )
    {
    }
    KSGenValueSet::KSGenValueSet( const KSGenValueSet& aCopy ) :
            KSComponent(),
            fValueStart( aCopy.fValueStart ),
            fValueStop( aCopy.fValueStop ),
            fValueCount( aCopy.fValueCount )
    {
    }
    KSGenValueSet* KSGenValueSet::Clone() const
    {
        return new KSGenValueSet( *this );
    }
    KSGenValueSet::~KSGenValueSet()
    {
    }

    void KSGenValueSet::DiceValue( vector< double >& aDicedValues )
    {
        double tValue;
        double tValueIncrement = (fValueStop - fValueStart) / ((double) (fValueCount > 1 ? fValueCount - 1 : 1));

        for( unsigned int tIndex = 0; tIndex < fValueCount; tIndex++ )
        {
            tValue = fValueStart + tIndex * tValueIncrement;
            aDicedValues.push_back( tValue );
        }
    }

}

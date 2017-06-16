//
// Created by trost on 07.03.16.
//

#ifndef KASPER_KSWRITEROOTCONDITIONTERMINATOR_H
#define KASPER_KSWRITEROOTCONDITIONTERMINATOR_H

#include "KSComponent.h"
#include "KSWriteROOTCondition.h"
#include "KField.h"

namespace Kassiopeia
{


class KSWriteROOTConditionTerminator :
    public KSComponentTemplate< KSWriteROOTConditionTerminator, KSWriteROOTCondition >
{
public:
    KSWriteROOTConditionTerminator() :
        fComponent( 0 ),
        fValue( 0 ),
        fMatchTerminator( std::string("") )
    {
    }
    KSWriteROOTConditionTerminator( const KSWriteROOTConditionTerminator& aCopy ) :
        KSComponent(aCopy),
        fComponent( aCopy.fComponent ),
        fValue( aCopy.fValue ),
        fMatchTerminator( aCopy.fMatchTerminator )
    {
    }
    KSWriteROOTConditionTerminator* Clone() const
    {
        return new KSWriteROOTConditionTerminator( *this );
    }
    virtual ~KSWriteROOTConditionTerminator()
    {
    }

    void CalculateWriteCondition( bool& aFlag )
    {
        aFlag = (fValue->compare(fMatchTerminator) == 0);

        return;
    }


protected:
    ;K_SET_GET_PTR(KSComponent, Component);
    ;K_SET_GET_PTR( std::string, Value );
    ;K_SET_GET( std::string, MatchTerminator );

};

}


#endif //KASPER_KSWRITEROOTCONDITIONTERMINATOR_H

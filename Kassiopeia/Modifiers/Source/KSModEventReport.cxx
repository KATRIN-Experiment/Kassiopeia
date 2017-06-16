#include "KSModEventReport.h"
#include "KSModifiersMessage.h"
#include <string>
#include <sstream>

namespace Kassiopeia
{

KSModEventReport::KSModEventReport(){};

KSModEventReport::KSModEventReport( const KSModEventReport& /*aCopy*/):KSComponent(){};


KSModEventReport*
KSModEventReport::Clone() const
{
    return new KSModEventReport(*this);
}

KSModEventReport::~KSModEventReport(){};

bool
KSModEventReport::ExecutePreEventModification( KSEvent& /*anEvent */ )
{
    modmsg( eNormal ) <<this->GetName() <<": reports that a new event is starting." << eom;
    return false;
};


bool
KSModEventReport::ExecutePostEventModification( KSEvent& /*anEvent*/ )
{
    modmsg( eNormal ) <<this->GetName() <<": reports that a event is complete." << eom;
    return false;
}


}

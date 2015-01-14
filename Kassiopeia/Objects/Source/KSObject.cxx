#include "KSObject.h"

namespace Kassiopeia
{

    KSObject::KSObject() :
            KTagged(),
            fHolder( NULL )
    {
    }
    KSObject::KSObject( const KSObject& aCopy ) :
            KTagged( aCopy ),
            fHolder( NULL )
    {
    }
    KSObject::~KSObject()
    {
    }

}

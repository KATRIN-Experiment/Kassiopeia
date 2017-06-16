#include "KApplication.h"

namespace katrin{

    KApplication::KApplication():
        KTagged()
    {
    }

    KApplication::KApplication(const KApplication &aCopy):
        KTagged( aCopy )
    {
    }

    KApplication::~KApplication()
    {
    }

}
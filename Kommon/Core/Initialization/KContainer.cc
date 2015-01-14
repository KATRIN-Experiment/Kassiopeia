#include "KContainer.hh"

namespace katrin
{

    KContainer::KContainer() :
        KNamed(),
        fHolder( NULL )
    {
    }
    KContainer::~KContainer()
    {
        if( fHolder != NULL )
        {
            delete fHolder;
            fHolder = NULL;
        }
    }

    bool KContainer::Empty() const
    {
        if( fHolder != NULL )
        {
            return false;
        }
        return true;
    }

}

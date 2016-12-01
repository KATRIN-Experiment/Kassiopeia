#ifndef Kommon_KElementProcessor_hh_
#define Kommon_KElementProcessor_hh_

#include "KComplexElement.hh"

namespace katrin
{

    typedef KComplexElement< void > KElementProcessor;

    template< >
    inline bool KElementProcessor::Begin()
    {
        return true;
    }

    template< >
    inline bool KElementProcessor::AddElement( KContainer* )
    {
        return true;
    }

    template< >
    inline bool KElementProcessor::AddAttribute( KContainer* )
    {
        return false;
    }

    template< >
    inline bool KElementProcessor::End()
    {
        return true;
    }
}

#endif

#ifndef KAPPLICATION_H_
#define KAPPLICATION_H_

#include "KTagged.h"

namespace katrin{

    class KApplication:
        public KTagged
    {
    public:
        KApplication();
        KApplication( const KApplication& aCopy);
        virtual ~KApplication();
        virtual bool Execute() = 0;

    };

}

#endif
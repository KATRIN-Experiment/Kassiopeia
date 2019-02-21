#ifndef KGBOUNDARY_HH_
#define KGBOUNDARY_HH_

#include "KGVisitor.hh"

#include "KTagged.h"
using katrin::KTagged;

namespace KGeoBag
{
    class KGBoundary :
        public KTagged
    {
        public:
            class Visitor {
            public:
                Visitor() {}
                virtual ~Visitor() {}
            };

        public:
            KGBoundary();
            KGBoundary( const KGBoundary& aBoundary );
            virtual ~KGBoundary();

        public:
            void Accept( KGVisitor* aVisitor );

        protected:
            mutable bool fInitialized;
    };

}

#endif

#ifndef KGeoBag_KGExtrudedPolyLoopSpaceMesher_hh_
#define KGeoBag_KGExtrudedPolyLoopSpaceMesher_hh_

#include "KGExtrudedPolyLoopSpace.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGExtrudedPolyLoopSpaceMesher :
        virtual public KGSimpleMesher,
        public KGExtrudedPolyLoopSpace::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGExtrudedPolyLoopSpaceMesher();
            virtual ~KGExtrudedPolyLoopSpaceMesher();

        protected:
            void VisitExtrudedClosedPathSpace( KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace );
    };

}

#endif

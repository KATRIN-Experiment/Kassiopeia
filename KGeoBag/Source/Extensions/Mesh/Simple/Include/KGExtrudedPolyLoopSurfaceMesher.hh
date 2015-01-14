#ifndef KGeoBag_KGExtrudedPolyLoopSurfaceMesher_hh_
#define KGeoBag_KGExtrudedPolyLoopSurfaceMesher_hh_

#include "KGExtrudedPolyLoopSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGExtrudedPolyLoopSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGExtrudedPolyLoopSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGExtrudedPolyLoopSurfaceMesher();
            virtual ~KGExtrudedPolyLoopSurfaceMesher();

        protected:
            void VisitExtrudedPathSurface( KGExtrudedPolyLoopSurface* aExtrudedPolyLoopSurface );
    };

}

#endif

#ifndef KGeoBag_KGExtrudedArcSegmentSurfaceMesher_hh_
#define KGeoBag_KGExtrudedArcSegmentSurfaceMesher_hh_

#include "KGExtrudedArcSegmentSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGExtrudedArcSegmentSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGExtrudedArcSegmentSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGExtrudedArcSegmentSurfaceMesher();
            virtual ~KGExtrudedArcSegmentSurfaceMesher();

        protected:
            void VisitExtrudedPathSurface( KGExtrudedArcSegmentSurface* aExtrudedArcSegmentSurface );
    };

}

#endif

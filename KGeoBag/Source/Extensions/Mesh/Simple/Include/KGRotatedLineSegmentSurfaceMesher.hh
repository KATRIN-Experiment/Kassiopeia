#ifndef KGeoBag_KGRotatedLineSegmentSurfaceMesher_hh_
#define KGeoBag_KGRotatedLineSegmentSurfaceMesher_hh_

#include "KGRotatedLineSegmentSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGRotatedLineSegmentSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGRotatedLineSegmentSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGRotatedLineSegmentSurfaceMesher();
            virtual ~KGRotatedLineSegmentSurfaceMesher();

        protected:
            void VisitRotatedPathSurface( KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface );
    };

}

#endif

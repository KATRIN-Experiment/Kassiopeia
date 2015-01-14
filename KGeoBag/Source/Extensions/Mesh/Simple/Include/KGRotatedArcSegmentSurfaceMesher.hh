#ifndef KGeoBag_KGRotatedArcSegmentSurfaceMesher_hh_
#define KGeoBag_KGRotatedArcSegmentSurfaceMesher_hh_

#include "KGRotatedArcSegmentSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGRotatedArcSegmentSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGRotatedArcSegmentSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGRotatedArcSegmentSurfaceMesher();
            virtual ~KGRotatedArcSegmentSurfaceMesher();

        protected:
            void VisitRotatedPathSurface( KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface );
    };

}

#endif

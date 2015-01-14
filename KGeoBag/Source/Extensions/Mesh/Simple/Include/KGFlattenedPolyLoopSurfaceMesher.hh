#ifndef KGeoBag_KGFlattenedPolyLoopSurfaceMesher_hh_
#define KGeoBag_KGFlattenedPolyLoopSurfaceMesher_hh_

#include "KGFlattenedPolyLoopSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGFlattenedPolyLoopSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGFlattenedPolyLoopSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGFlattenedPolyLoopSurfaceMesher();
            virtual ~KGFlattenedPolyLoopSurfaceMesher();

        protected:
            void VisitFlattenedClosedPathSurface( KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface );
    };

}

#endif

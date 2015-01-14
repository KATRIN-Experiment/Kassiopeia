#ifndef KGeoBag_KGShellLineSegmentSurfaceMesher_hh_
#define KGeoBag_KGShellLineSegmentSurfaceMesher_hh_

#include "KGShellLineSegmentSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGShellLineSegmentSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGShellLineSegmentSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGShellLineSegmentSurfaceMesher();
            virtual ~KGShellLineSegmentSurfaceMesher();

        protected:
            void VisitShellPathSurface( KGShellLineSegmentSurface* aShellLineSegmentSurface );
    };

}

#endif

#ifndef KGeoBag_KGExtrudedPolyLineSurfaceMesher_hh_
#define KGeoBag_KGExtrudedPolyLineSurfaceMesher_hh_

#include "KGExtrudedPolyLineSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGExtrudedPolyLineSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGExtrudedPolyLineSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGExtrudedPolyLineSurfaceMesher();
            virtual ~KGExtrudedPolyLineSurfaceMesher();

        protected:
            void VisitExtrudedPathSurface( KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface );
    };

}

#endif

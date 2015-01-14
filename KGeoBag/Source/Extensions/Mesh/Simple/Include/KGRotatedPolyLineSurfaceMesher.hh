#ifndef KGeoBag_KGRotatedPolyLineSurfaceMesher_hh_
#define KGeoBag_KGRotatedPolyLineSurfaceMesher_hh_

#include "KGRotatedPolyLineSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGRotatedPolyLineSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGRotatedPolyLineSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGRotatedPolyLineSurfaceMesher();
            virtual ~KGRotatedPolyLineSurfaceMesher();

        protected:
            void VisitRotatedPathSurface( KGRotatedPolyLineSurface* aRotatedPolyLineSurface );
    };

}

#endif

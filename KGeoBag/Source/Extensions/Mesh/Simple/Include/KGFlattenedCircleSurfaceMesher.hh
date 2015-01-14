#ifndef KGeoBag_KGFlattenedCircleSurfaceMesher_hh_
#define KGeoBag_KGFlattenedCircleSurfaceMesher_hh_

#include "KGFlattenedCircleSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGFlattenedCircleSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGFlattenedCircleSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGFlattenedCircleSurfaceMesher();
            virtual ~KGFlattenedCircleSurfaceMesher();

        protected:
            void VisitFlattenedClosedPathSurface( KGFlattenedCircleSurface* aFlattenedCircleSurface );
    };

}

#endif

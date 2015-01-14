#ifndef KGeoBag_KGRotatedPolyLoopSurfaceAxialMesher_hh_
#define KGeoBag_KGRotatedPolyLoopSurfaceAxialMesher_hh_

#include "KGRotatedPolyLoopSurface.hh"

#include "KGSimpleAxialMesher.hh"

namespace KGeoBag
{
    class KGRotatedPolyLoopSurfaceAxialMesher :
        virtual public KGSimpleAxialMesher,
        public KGRotatedPolyLoopSurface::Visitor
    {
        public:
            using KGAxialMesherBase::VisitExtendedSurface;
            using KGAxialMesherBase::VisitExtendedSpace;

        public:
            KGRotatedPolyLoopSurfaceAxialMesher();
            virtual ~KGRotatedPolyLoopSurfaceAxialMesher();

        protected:
            void VisitRotatedPathSurface( KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface );
    };

}

#endif

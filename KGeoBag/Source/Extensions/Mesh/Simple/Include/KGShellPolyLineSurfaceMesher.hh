#ifndef KGeoBag_KGShellPolyLineSurfaceMesher_hh_
#define KGeoBag_KGShellPolyLineSurfaceMesher_hh_

#include "KGShellPolyLineSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGShellPolyLineSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGShellPolyLineSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGShellPolyLineSurfaceMesher();
            virtual ~KGShellPolyLineSurfaceMesher();

        protected:
            void VisitShellPathSurface( KGShellPolyLineSurface* aShellPolyLineSurface );
    };

}

#endif

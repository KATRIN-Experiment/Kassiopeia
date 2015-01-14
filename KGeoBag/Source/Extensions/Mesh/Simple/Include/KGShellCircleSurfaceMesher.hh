#ifndef KGeoBag_KGShellCircleSurfaceMesher_hh_
#define KGeoBag_KGShellCircleSurfaceMesher_hh_

#include "KGShellCircleSurface.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGShellCircleSurfaceMesher :
        virtual public KGSimpleMesher,
        public KGShellCircleSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGShellCircleSurfaceMesher();
            virtual ~KGShellCircleSurfaceMesher();

        protected:
            void VisitShellPathSurface( KGShellCircleSurface* aShellCircleSurface );
    };

}

#endif

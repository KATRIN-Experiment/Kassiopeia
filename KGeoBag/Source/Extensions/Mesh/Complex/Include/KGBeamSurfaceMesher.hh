#ifndef KGeoBag_KGBeamSurfaceMesher_hh_
#define KGeoBag_KGBeamSurfaceMesher_hh_

#include "KGBeamSurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
    class KGBeamSurfaceMesher :
        virtual public KGComplexMesher,
        public KGBeamSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGBeamSurfaceMesher()
            {
            }
            virtual ~KGBeamSurfaceMesher()
            {
            }

        protected:
            void VisitWrappedSurface( KGBeamSurface* beamSurface );

    };
}

#endif /* KGBEAMSURFACEMESHER_HH_ */

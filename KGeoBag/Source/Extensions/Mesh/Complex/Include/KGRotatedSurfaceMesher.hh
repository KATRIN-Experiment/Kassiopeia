#ifndef KGeoBag_KGRotatedSurfaceMesher_hh_
#define KGeoBag_KGRotatedSurfaceMesher_hh_

#include "KGRotatedSurface.hh"

#include "KGComplexMesher.hh"

namespace KGeoBag
{
    class KGRotatedSurfaceMesher :
        virtual public KGComplexMesher,
        public KGRotatedSurface::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGRotatedSurfaceMesher()
            {
            }
            virtual ~KGRotatedSurfaceMesher()
            {
            }

        protected:
            void VisitWrappedSurface( KGRotatedSurface* rotatedSurface );

            void DiscretizeSegment( const KGRotatedObject::Line* line, const unsigned int nPolyBegin, const unsigned int nPolyEnd );
            void DiscretizeSegment( const KGRotatedObject::Arc* arc, const unsigned int nPolyBegin, const unsigned int nPolyEnd );
    };
}

#endif /* KGROTATEDSURFACEDISCRETIZER_DEF */

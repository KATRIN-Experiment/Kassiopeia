#ifndef KGeoBag_KGRotatedPolyLoopSpaceMesher_hh_
#define KGeoBag_KGRotatedPolyLoopSpaceMesher_hh_

#include "KGRotatedPolyLoopSpace.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGRotatedPolyLoopSpaceMesher :
        virtual public KGSimpleMesher,
        public KGRotatedPolyLoopSpace::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGRotatedPolyLoopSpaceMesher();
            virtual ~KGRotatedPolyLoopSpaceMesher();

        protected:
            void VisitRotatedClosedPathSpace( KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace );
    };

}

#endif

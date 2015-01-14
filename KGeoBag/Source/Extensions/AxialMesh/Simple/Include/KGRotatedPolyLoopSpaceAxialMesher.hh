#ifndef KGeoBag_KGRotatedPolyLoopSpaceAxialMesher_hh_
#define KGeoBag_KGRotatedPolyLoopSpaceAxialMesher_hh_

#include "KGRotatedPolyLoopSpace.hh"

#include "KGSimpleAxialMesher.hh"

namespace KGeoBag
{
    class KGRotatedPolyLoopSpaceAxialMesher :
        virtual public KGSimpleAxialMesher,
        public KGRotatedPolyLoopSpace::Visitor
    {
        public:
            using KGAxialMesherBase::VisitExtendedSurface;
            using KGAxialMesherBase::VisitExtendedSpace;

        public:
            KGRotatedPolyLoopSpaceAxialMesher();
            virtual ~KGRotatedPolyLoopSpaceAxialMesher();

        protected:
            void VisitRotatedClosedPathSpace( KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace );
    };

}

#endif

#ifndef KGeoBag_KGRotatedCircleSpaceAxialMesher_hh_
#define KGeoBag_KGRotatedCircleSpaceAxialMesher_hh_

#include "KGRotatedCircleSpace.hh"

#include "KGSimpleAxialMesher.hh"

namespace KGeoBag
{
    class KGRotatedCircleSpaceAxialMesher :
        virtual public KGSimpleAxialMesher,
        public KGRotatedCircleSpace::Visitor
    {
        public:
            using KGAxialMesherBase::VisitExtendedSurface;
            using KGAxialMesherBase::VisitExtendedSpace;

        public:
            KGRotatedCircleSpaceAxialMesher();
            virtual ~KGRotatedCircleSpaceAxialMesher();

        protected:
            void VisitRotatedClosedPathSpace( KGRotatedCircleSpace* aRotatedCircleSpace );
    };

}

#endif

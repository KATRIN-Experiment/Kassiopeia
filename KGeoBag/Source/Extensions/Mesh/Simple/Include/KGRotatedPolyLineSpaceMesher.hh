#ifndef KGeoBag_KGRotatedPolyLineSpaceMesher_hh_
#define KGeoBag_KGRotatedPolyLineSpaceMesher_hh_

#include "KGRotatedPolyLineSpace.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGRotatedPolyLineSpaceMesher :
        virtual public KGSimpleMesher,
        public KGRotatedPolyLineSpace::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGRotatedPolyLineSpaceMesher();
            virtual ~KGRotatedPolyLineSpaceMesher();

        protected:
            void VisitRotatedOpenPathSpace( KGRotatedPolyLineSpace* aRotatedPolyLineSpace );
    };

}

#endif

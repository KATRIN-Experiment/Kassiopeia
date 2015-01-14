#ifndef KGeoBag_KGExtrudedCircleSpaceMesher_hh_
#define KGeoBag_KGExtrudedCircleSpaceMesher_hh_

#include "KGExtrudedCircleSpace.hh"

#include "KGSimpleMesher.hh"

namespace KGeoBag
{
    class KGExtrudedCircleSpaceMesher :
        virtual public KGSimpleMesher,
        public KGExtrudedCircleSpace::Visitor
    {
        public:
            using KGMesherBase::VisitExtendedSurface;
            using KGMesherBase::VisitExtendedSpace;

        public:
            KGExtrudedCircleSpaceMesher();
            virtual ~KGExtrudedCircleSpaceMesher();

        protected:
            void VisitExtrudedClosedPathSpace( KGExtrudedCircleSpace* aExtrudedCircleSpace );
    };

}

#endif

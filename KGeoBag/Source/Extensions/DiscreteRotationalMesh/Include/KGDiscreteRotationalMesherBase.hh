#ifndef KGeoBag_KGDiscreteRotationalMesherBase_hh_
#define KGeoBag_KGDiscreteRotationalMesherBase_hh_

#include "KGCore.hh"

#include "KGDiscreteRotationalMesh.hh"

namespace KGeoBag
{

    class KGDiscreteRotationalMesherBase :
        public KGVisitor,
        public KGExtendedSurface< KGDiscreteRotationalMesh >::Visitor,
        public KGExtendedSpace< KGDiscreteRotationalMesh >::Visitor
    {
        protected:
            KGDiscreteRotationalMesherBase();

        public:
            virtual ~KGDiscreteRotationalMesherBase();

        public:
            virtual void VisitExtendedSurface( KGExtendedSurface< KGDiscreteRotationalMesh >* aSurface );
            virtual void VisitExtendedSpace( KGExtendedSpace< KGDiscreteRotationalMesh >* aSpace );

        protected:
            KGDiscreteRotationalMeshElementVector* fCurrentElements;
    };

}

#endif

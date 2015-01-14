#ifndef KGeoBag_KGAxialMesherBase_hh_
#define KGeoBag_KGAxialMesherBase_hh_

#include "KGCore.hh"

#include "KGAxialMesh.hh"

namespace KGeoBag
{

    class KGAxialMesherBase :
        public KGVisitor,
        public KGExtendedSurface< KGAxialMesh >::Visitor,
        public KGExtendedSpace< KGAxialMesh >::Visitor
    {
        protected:
            KGAxialMesherBase();

        public:
            virtual ~KGAxialMesherBase();

        public:
            virtual void VisitExtendedSurface( KGExtendedSurface< KGAxialMesh >* aSurface );
            virtual void VisitExtendedSpace( KGExtendedSpace< KGAxialMesh >* aSpace );

        protected:
            KGAxialMeshElementVector* fCurrentElements;
    };

}

#endif

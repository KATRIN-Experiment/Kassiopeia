#ifndef KGeoBag_KGDiscreteRotationalMesherBase_hh_
#define KGeoBag_KGDiscreteRotationalMesherBase_hh_

#include "KGCore.hh"

#include "KGDiscreteRotationalMesh.hh"

namespace KGeoBag
{

    class KGDiscreteRotationalMesherBase :
    	public KGVisitor
    {
        protected:
            KGDiscreteRotationalMesherBase();

        public:
            virtual ~KGDiscreteRotationalMesherBase();

        public:
            void SetMeshElementOutput(KGDiscreteRotationalMeshElementVector* aMeshElementVector );
            KGDiscreteRotationalMeshElementVector* GetCurrentElements() const {return fCurrentElements;}

        protected:
            KGDiscreteRotationalMeshElementVector* fCurrentElements;
    };

}

#endif

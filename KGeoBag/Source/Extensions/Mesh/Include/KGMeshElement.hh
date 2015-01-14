#ifndef KGeoBag_KGMeshElement_hh_
#define KGeoBag_KGMeshElement_hh_

#include "KTransformation.hh"

#include <vector>
using std::vector;

namespace KGeoBag
{

    class KGMeshElement
    {
        public:
            KGMeshElement();
            virtual ~KGMeshElement();

            virtual double Area() const = 0;
            virtual double Aspect() const = 0;

            virtual void Transform( const KTransformation& transform ) = 0;
    };

    typedef vector< KGMeshElement* > KGMeshElementVector;
    typedef vector< KGMeshElement* >::iterator KGMeshElementIt;
    typedef vector< KGMeshElement* >::const_iterator KGMeshElementCIt;

}

#endif

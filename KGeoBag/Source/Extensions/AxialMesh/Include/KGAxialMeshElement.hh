#ifndef KGeoBag_KGAxialMeshElement_hh_
#define KGeoBag_KGAxialMeshElement_hh_

#include <vector>
using std::vector;

namespace KGeoBag
{

class KGAxialMeshElement
{
  public:
    KGAxialMeshElement();
    virtual ~KGAxialMeshElement();

    virtual double Area() const = 0;
    virtual double Aspect() const = 0;
};

typedef vector<KGAxialMeshElement*> KGAxialMeshElementVector;
typedef vector<KGAxialMeshElement*>::iterator KGAxialMeshElementIt;
typedef vector<KGAxialMeshElement*>::const_iterator KGAxialMeshElementCIt;

}  // namespace KGeoBag

#endif

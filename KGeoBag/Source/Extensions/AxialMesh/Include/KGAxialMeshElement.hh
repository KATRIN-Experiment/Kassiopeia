#ifndef KGeoBag_KGAxialMeshElement_hh_
#define KGeoBag_KGAxialMeshElement_hh_

#include <string>
#include <vector>

namespace KGeoBag
{

class KGAxialMeshElement
{
  public:
    KGAxialMeshElement();
    virtual ~KGAxialMeshElement();

    static std::string Name()
    {
        return "axial_mesh_base";
    }

    virtual double Area() const = 0;
    virtual double Aspect() const = 0;
};

typedef std::vector<KGAxialMeshElement*> KGAxialMeshElementVector;
using KGAxialMeshElementIt = KGAxialMeshElementVector::iterator;
using KGAxialMeshElementCIt = KGAxialMeshElementVector::const_iterator;

}  // namespace KGeoBag

#endif

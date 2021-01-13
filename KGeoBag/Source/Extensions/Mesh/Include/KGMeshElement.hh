#ifndef KGeoBag_KGMeshElement_hh_
#define KGeoBag_KGMeshElement_hh_

#include "KGPointCloud.hh"
#include "KTransformation.hh"

#include <string>
#include <vector>

#define KGMESH_DIM 3

namespace KGeoBag
{

class KGMeshElement
{
  public:
    KGMeshElement();
    virtual ~KGMeshElement();

    static std::string Name()
    {
        return "mesh_base";
    }

    virtual double Area() const = 0;
    virtual double Aspect() const = 0;

    virtual void Transform(const KTransformation& transform) = 0;

    virtual double NearestDistance(const KGeoBag::KThreeVector& /*aPoint*/) const = 0;
    virtual KGeoBag::KThreeVector NearestPoint(const KGeoBag::KThreeVector& /*aPoint*/) const = 0;
    virtual KGeoBag::KThreeVector NearestNormal(const KGeoBag::KThreeVector& /*aPoint*/) const = 0;
    virtual bool NearestIntersection(const KGeoBag::KThreeVector& /*aStart*/, const KGeoBag::KThreeVector& /*anEnd*/,
                                     KGeoBag::KThreeVector& /*anIntersection*/) const = 0;

    virtual KGPointCloud<KGMESH_DIM> GetPointCloud() const = 0;
    virtual unsigned int GetNumberOfEdges() const = 0;
    virtual void GetEdge(KThreeVector& start, KGeoBag::KThreeVector& end, unsigned int /*index*/) const = 0;
};

typedef std::vector<KGMeshElement*> KGMeshElementVector;
using KGMeshElementIt = KGMeshElementVector::iterator;
using KGMeshElementCIt = KGMeshElementVector::const_iterator;

}  // namespace KGeoBag

#endif

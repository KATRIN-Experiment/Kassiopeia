#ifndef KGeoBag_KGMeshElement_hh_
#define KGeoBag_KGMeshElement_hh_

#include "KGPointCloud.hh"

#include "KTransformation.hh"
#include "KThreeVector.hh"

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
    virtual katrin::KThreeVector Centroid() const = 0;

    virtual void Transform(const katrin::KTransformation& transform) = 0;

    virtual double NearestDistance(const katrin::KThreeVector& /*aPoint*/) const = 0;
    virtual katrin::KThreeVector NearestPoint(const katrin::KThreeVector& /*aPoint*/) const = 0;
    virtual katrin::KThreeVector NearestNormal(const katrin::KThreeVector& /*aPoint*/) const = 0;
    virtual bool NearestIntersection(const katrin::KThreeVector& /*aStart*/, const katrin::KThreeVector& /*anEnd*/,
                                     katrin::KThreeVector& /*anIntersection*/) const = 0;

    virtual KGPointCloud<KGMESH_DIM> GetPointCloud() const = 0;
    virtual unsigned int GetNumberOfEdges() const = 0;
    virtual void GetEdge(katrin::KThreeVector& start, katrin::KThreeVector& end, unsigned int /*index*/) const = 0;
};

typedef std::vector<KGMeshElement*> KGMeshElementVector;
using KGMeshElementIt = KGMeshElementVector::iterator;
using KGMeshElementCIt = KGMeshElementVector::const_iterator;

}  // namespace KGeoBag

#endif

#ifndef KGeoBag_KGMeshElement_hh_
#define KGeoBag_KGMeshElement_hh_

#include "KGPointCloud.hh"
#include "KTransformation.hh"

#include <vector>
using std::vector;

#define KGMESH_DIM 3

namespace KGeoBag
{

class KGMeshElement
{
  public:
    KGMeshElement();
    virtual ~KGMeshElement();

    virtual double Area() const = 0;
    virtual double Aspect() const = 0;

    virtual void Transform(const KTransformation& transform) = 0;

    virtual double NearestDistance(const KThreeVector& /*aPoint*/) const = 0;
    virtual KThreeVector NearestPoint(const KThreeVector& /*aPoint*/) const = 0;
    virtual KThreeVector NearestNormal(const KThreeVector& /*aPoint*/) const = 0;
    virtual bool NearestIntersection(const KThreeVector& /*aStart*/, const KThreeVector& /*anEnd*/,
                                     KThreeVector& /*anIntersection*/) const = 0;

    virtual KGPointCloud<KGMESH_DIM> GetPointCloud() const = 0;
    virtual unsigned int GetNumberOfEdges() const = 0;
    virtual void GetEdge(KThreeVector& start, KThreeVector& end, unsigned int /*index*/) const = 0;
};

typedef vector<KGMeshElement*> KGMeshElementVector;
typedef vector<KGMeshElement*>::iterator KGMeshElementIt;
typedef vector<KGMeshElement*>::const_iterator KGMeshElementCIt;

}  // namespace KGeoBag

#endif

#ifndef KGMeshRefiner_DEF
#define KGMeshRefiner_DEF

#include "KGCore.hh"
#include "KGMesh.hh"

namespace KGeoBag
{
class KGMeshElement;
class KGMeshRectangle;
class KGMeshTriangle;
class KGMeshWire;

class KGMeshRefiner :
    public KGVisitor,
    public KGExtendedSurface<KGMesh>::Visitor
{
  public:
    KGMeshRefiner();
    ~KGMeshRefiner() override = default;

    void SetMaximumNumberOfRefinements(int steps) { fMaxNumRefinements = steps; }
    void SetMaximumLength(double length) { fMaxLength = length; }
    void SetMaximumArea(double area) { fMaxArea = area; }
    void SetMaximumAspectRatio(double aspect) { fMaxAspect = aspect; }

    void VisitSurface(KGSurface*) {}
    void VisitExtendedSurface(KGExtendedSurface<KGMesh>* meshSurface) override;

  private:
    void AddRefined(KGMeshElement* e, int maxDepth);
    void AddRefined(KGMeshRectangle* r, int maxDepth);
    void AddRefined(KGMeshTriangle* t, int maxDepth);
    void AddRefined(KGMeshWire* w, int maxDepth);

    double fMaxLength;
    double fMaxArea;
    double fMaxAspect;
    int fMaxNumRefinements;
    KGMeshElementVector fRefinedVector;
};
}  // namespace KGeoBag

#endif /* KGMeshRefiner_DEF */

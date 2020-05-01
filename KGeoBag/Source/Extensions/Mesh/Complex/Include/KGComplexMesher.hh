#ifndef KGeoBag_KGComplexMesher_hh_
#define KGeoBag_KGComplexMesher_hh_

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGMesherBase.hh"

namespace KGeoBag
{

class KGComplexMesher : virtual public KGMesherBase
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  protected:
    KGComplexMesher();

  public:
    ~KGComplexMesher() override;

    static void DiscretizeInterval(double interval, int nSegments, double power, std::vector<double>& segments);

  protected:
    void AddElement(KGMeshElement* e);
    void RefineAndAddElement(KGMeshRectangle* rectangle, int nElements_A, double power_A, int nElements_B,
                             double power_B);
    void RefineAndAddElement(KGMeshTriangle* triangle, int nElements, double power);
    void RefineAndAddElement(KGMeshWire* wire, int nElements, double power);
};
}  // namespace KGeoBag

#endif

#ifndef KGeoBag_KGExtrudedSurfaceMesher_hh_
#define KGeoBag_KGExtrudedSurfaceMesher_hh_

#include "KGComplexMesher.hh"
#include "KGExtrudedSurface.hh"

namespace KGeoBag
{
class KGExtrudedSurfaceMesher : virtual public KGComplexMesher, public KGExtrudedSurface::Visitor
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGExtrudedSurfaceMesher() : fExtrudedObject(nullptr), fIsModifiable(false) {}
    ~KGExtrudedSurfaceMesher() override = default;

  protected:
    void VisitWrappedSurface(KGExtrudedSurface* extrudedSurface) override;

    void Discretize(KGExtrudedObject* object);
    void DiscretizeSegment(const KGExtrudedObject::Line* line, const unsigned int nDisc,
                           std::vector<std::vector<double>>& coords, unsigned int& counter);
    void DiscretizeSegment(const KGExtrudedObject::Arc* arc, const unsigned int nDisc,
                           std::vector<std::vector<double>>& coords, unsigned int& counter);
    void DiscretizeEnclosedEnds(std::vector<std::vector<double>>& iCoords, std::vector<std::vector<double>>& oCoords,
                                unsigned int nDisc);
    void DiscretizeLoopEnds();

    virtual void ModifyInnerSegment(int, std::vector<std::vector<double>>&) {}
    virtual void ModifyOuterSegment(int, std::vector<std::vector<double>>&) {}
    virtual void ModifySurface(std::vector<std::vector<double>>&, std::vector<std::vector<double>>&,
                               std::vector<unsigned int>&, std::vector<unsigned int>&)
    {}

    KGExtrudedObject* fExtrudedObject;

    bool fIsModifiable;
};
}  // namespace KGeoBag

#endif /* KGEXTRUDEDSURFACEDISCRETIZER_DEF */

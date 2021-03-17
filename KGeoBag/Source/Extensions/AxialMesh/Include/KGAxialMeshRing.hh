#ifndef KGeoBag_KGAxialMeshRing_hh_
#define KGeoBag_KGAxialMeshRing_hh_

#include "KGAxialMeshElement.hh"
#include "KTwoVector.hh"

namespace KGeoBag
{

class KGAxialMeshRing : public KGAxialMeshElement
{
  public:
    KGAxialMeshRing(const double& aD, const KTwoVector& aP0);
    ~KGAxialMeshRing() override;

    static std::string Name()
    {
        return "axial_mesh_ring";
    }

    double Area() const override;
    double Aspect() const override;

    const double& GetD() const
    {
        return fD;
    }
    const KTwoVector& GetP0() const
    {
        return fP0;
    }
    void GetP0(KTwoVector& aP0) const
    {
        aP0 = fP0;
        return;
    }

  private:
    double fD;
    KTwoVector fP0;
};

}  // namespace KGeoBag

#endif

#ifndef KGeoBag_KGAxialMeshLoop_hh_
#define KGeoBag_KGAxialMeshLoop_hh_

#include "KGAxialMeshElement.hh"

#include "KTwoVector.hh"

namespace KGeoBag
{

class KGAxialMeshLoop : public KGAxialMeshElement
{
  public:
    KGAxialMeshLoop(const katrin::KTwoVector& aStart, const katrin::KTwoVector& p1);
    ~KGAxialMeshLoop() override;

    static std::string Name()
    {
        return "axial_mesh_loop";
    }

    double Area() const override;
    double Aspect() const override;

    const katrin::KTwoVector& GetP0() const
    {
        return fP0;
    }
    const katrin::KTwoVector& GetP1() const
    {
        return fP1;
    }
    void GetP0(katrin::KTwoVector& aP0) const
    {
        aP0 = fP0;
        return;
    }
    void GetP1(katrin::KTwoVector& aP1) const
    {
        aP1 = fP1;
        return;
    }

  private:
    katrin::KTwoVector fP0;
    katrin::KTwoVector fP1;
};

}  // namespace KGeoBag

#endif

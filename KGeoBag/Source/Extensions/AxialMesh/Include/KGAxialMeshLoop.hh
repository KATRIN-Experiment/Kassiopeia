#ifndef KGeoBag_KGAxialMeshLoop_hh_
#define KGeoBag_KGAxialMeshLoop_hh_

#include "KGAxialMeshElement.hh"
#include "KTwoVector.hh"

namespace KGeoBag
{

class KGAxialMeshLoop : public KGAxialMeshElement
{
  public:
    KGAxialMeshLoop(const KTwoVector& aStart, const KTwoVector& p1);
    ~KGAxialMeshLoop() override;

    double Area() const override;
    double Aspect() const override;

    const KTwoVector& GetP0() const
    {
        return fP0;
    }
    const KTwoVector& GetP1() const
    {
        return fP1;
    }
    void GetP0(KTwoVector& aP0) const
    {
        aP0 = fP0;
        return;
    }
    void GetP1(KTwoVector& aP1) const
    {
        aP1 = fP1;
        return;
    }

  private:
    KTwoVector fP0;
    KTwoVector fP1;
};

}  // namespace KGeoBag

#endif

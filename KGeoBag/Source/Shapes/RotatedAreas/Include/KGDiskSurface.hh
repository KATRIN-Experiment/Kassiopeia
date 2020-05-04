#ifndef KGDISKSURFACE_HH_
#define KGDISKSURFACE_HH_

#include "KGRotatedLineSegmentSurface.hh"

namespace KGeoBag
{

class KGDiskSurface : public KGRotatedLineSegmentSurface
{
  public:
    KGDiskSurface();
    ~KGDiskSurface() override;

  public:
    void Z(const double& aZ);
    void R(const double& anR);
    void RadialMeshCount(const unsigned int& aRadialMeshCount);
    void RadialMeshPower(const double& aRadialMeshPower);
    void AxialMeshCount(const unsigned int& anAxialMeshCount);

    const double& Z() const;
    const double& R() const;
    const unsigned int& RadialMeshCount() const;
    const double& RadialMeshPower() const;
    const unsigned int& AxialMeshCount() const;

  private:
    double fZ;
    double fR;
    unsigned int fRadialMeshCount;
    double fRadialMeshPower;
    unsigned int fAxialMeshCount;

  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitDiskSurface(KGDiskSurface* aDisk) = 0;
    };

  public:
    void AreaInitialize() const override;
    void AreaAccept(KGVisitor* aVisitor) override;
};

}  // namespace KGeoBag

#endif

#ifndef KGCUTTORUSSURFACE_H_
#define KGCUTTORUSSURFACE_H_

#include "KGRotatedArcSegmentSurface.hh"

namespace KGeoBag
{

class KGCutTorusSurface : public KGRotatedArcSegmentSurface
{
  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitCutTorusSurface(KGCutTorusSurface* aCutTorusSurface) = 0;
    };

  public:
    KGCutTorusSurface();
    KGCutTorusSurface(const KGCutTorusSurface& aCopy);
    ~KGCutTorusSurface() override;

  public:
    void Z1(const double& aZ1);
    void Z2(const double& aZ2);
    void R1(const double& anR1);
    void R2(const double& anR2);
    void Radius(const double& aRadius);
    void Right(const bool& aRight);
    void Short(const bool& aShort);
    void ToroidalMeshCount(const unsigned int& aToroidalMeshCount);
    void AxialMeshCount(const unsigned int& anAxialMeshCount);

    const double& Z1() const;
    const double& R1() const;
    const double& Z2() const;
    const double& R2() const;
    const double& Radius() const;
    const bool& Right() const;
    const bool& Short() const;
    const unsigned int& ToroidalMeshCount() const;
    const unsigned int& AxialMeshCount() const;

  private:
    double fZ1;
    double fR1;
    double fZ2;
    double fR2;
    double fRadius;
    bool fRight;
    bool fShort;
    unsigned int fToroidalMeshCount;
    unsigned int fAxialMeshCount;

  public:
    void AreaInitialize() const override;
    void AreaAccept(KGVisitor* aVisitor) override;
};

}  // namespace KGeoBag

#endif

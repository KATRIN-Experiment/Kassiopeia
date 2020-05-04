#ifndef KGBOZSPACE_HH_
#define KGBOZSPACE_HH_

#include "KGExtrudedPolyLoopSpace.hh"

namespace KGeoBag
{

class KGBoxSpace : public KGExtrudedPolyLoopSpace
{
  public:
    KGBoxSpace();
    ~KGBoxSpace() override;

  public:
    void XA(const double& aXA);
    void XB(const double& aXB);
    void XMeshCount(const unsigned int& aXMeshCount);
    void XMeshPower(const double& aXMeshPower);

    void YA(const double& aYA);
    void YB(const double& aYB);
    void YMeshCount(const unsigned int& aYMeshCount);
    void YMeshPower(const double& aYMeshPower);

    void ZA(const double& aZA);
    void ZB(const double& aZB);
    void ZMeshCount(const unsigned int& aZMeshCount);
    void ZMeshPower(const double& aZMeshPower);

    const double& XA() const;
    const double& XB() const;
    const unsigned int& XMeshCount() const;
    const double& XMeshPower() const;

    const double& YA() const;
    const double& YB() const;
    const unsigned int& YMeshCount() const;
    const double& YMeshPower() const;

    const double& ZA() const;
    const double& ZB() const;
    const unsigned int& ZMeshCount() const;
    const double& ZMeshPower() const;

  private:
    double fXA;
    double fXB;
    unsigned int fXMeshCount;
    double fXMeshPower;

    double fYA;
    double fYB;
    unsigned int fYMeshCount;
    double fYMeshPower;

    double fZA;
    double fZB;
    unsigned int fZMeshCount;
    double fZMeshPower;

  public:
    class Visitor
    {
      public:
        Visitor();
        virtual ~Visitor();

        virtual void VisitBoxSpace(const KGBoxSpace* aBoxSpace) = 0;
    };

  public:
    void VolumeInitialize(BoundaryContainer& aBoundaryContainer) const override;
    void VolumeAccept(KGVisitor* aVisitor) override;
};

}  // namespace KGeoBag
#endif

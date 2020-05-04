#ifndef KGPLANARLINESEGMENT_HH_
#define KGPLANARLINESEGMENT_HH_

#include "KGPlanarOpenPath.hh"

namespace KGeoBag
{

class KGPlanarLineSegment : public KGPlanarOpenPath
{

  public:
    KGPlanarLineSegment();
    KGPlanarLineSegment(const KGPlanarLineSegment& aCopy);
    KGPlanarLineSegment(const KTwoVector& aStart, const KTwoVector& anEnd, const unsigned int aCount = 2,
                        const double aPower = 1.);
    KGPlanarLineSegment(const double& anX1, const double& aY1, const double& anX2, const double& aY2,
                        const unsigned int aCount = 2, const double aPower = 1.);
    ~KGPlanarLineSegment() override;

    KGPlanarLineSegment* Clone() const override;
    void CopyFrom(const KGPlanarLineSegment& aCopy);

  public:
    void Start(const KTwoVector& aStart);
    void X1(const double& aValue);
    void Y1(const double& aValue);
    void End(const KTwoVector& anEnd);
    void X2(const double& aValue);
    void Y2(const double& aValue);
    void MeshCount(const unsigned int& aCount);
    void MeshPower(const double& aPower);

    const KTwoVector& Start() const override;
    const double& X1() const;
    const double& Y1() const;
    const KTwoVector& End() const override;
    const double& X2() const;
    const double& Y2() const;
    const unsigned int& MeshCount() const;
    const double& MeshPower() const;

    const double& Length() const override;
    const KTwoVector& Centroid() const override;
    const KTwoVector& XUnit() const;
    const KTwoVector& YUnit() const;

  public:
    KTwoVector At(const double& aLength) const override;
    KTwoVector Point(const KTwoVector& aQuery) const override;
    KTwoVector Normal(const KTwoVector& aQuery) const override;
    bool Above(const KTwoVector& aQuery) const override;

  private:
    KTwoVector fStart;
    KTwoVector fEnd;
    unsigned int fMeshCount;
    double fMeshPower;

    mutable double fLength;
    mutable KTwoVector fCentroid;
    mutable KTwoVector fXUnit;
    mutable KTwoVector fYUnit;

    void Initialize() const;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif

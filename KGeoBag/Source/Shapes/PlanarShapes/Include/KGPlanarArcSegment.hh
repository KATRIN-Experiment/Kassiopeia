#ifndef KGPLANARARCSEGMENT_HH_
#define KGPLANARARCSEGMENT_HH_

#include "KGPlanarOpenPath.hh"

namespace KGeoBag
{

class KGPlanarArcSegment : public KGPlanarOpenPath
{
  public:
    KGPlanarArcSegment();
    KGPlanarArcSegment(const KGPlanarArcSegment& aCopy);
    KGPlanarArcSegment(const KTwoVector& aStart, const KTwoVector& anEnd, const double& aRadius, const bool& isRight,
                       const bool& isShort, const unsigned int aCount = 16);
    KGPlanarArcSegment(const double& anX1, const double& aY1, const double& anX2, const double& aY2,
                       const double& aRadius, const bool& isRight, const bool& isShort, const unsigned int aCount = 16);
    ~KGPlanarArcSegment() override;

    KGPlanarArcSegment* Clone() const override;
    void CopyFrom(const KGPlanarArcSegment& aCopy);

  public:
    void Start(const KTwoVector& aStart);
    void X1(const double& aValue);
    void Y1(const double& aValue);
    void End(const KTwoVector& anEnd);
    void X2(const double& aValue);
    void Y2(const double& aValue);
    void Radius(const double& aValue);
    void Right(const bool& aValue);
    void Short(const bool& aValue);
    void MeshCount(const unsigned int& aCount);

    const KTwoVector& Start() const override;
    const double& X1() const;
    const double& Y1() const;
    const KTwoVector& End() const override;
    const double& X2() const;
    const double& Y2() const;
    const double& Radius() const;
    const bool& Right() const;
    const bool& Short() const;
    const unsigned int& MeshCount() const;

    const double& Length() const override;
    const double& Angle() const;
    const KTwoVector& Centroid() const override;
    const KTwoVector& Origin() const;
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
    double fRadius;
    bool fRight;
    bool fShort;
    unsigned int fMeshCount;

    mutable double fLength;
    mutable double fAngle;
    mutable KTwoVector fCentroid;
    mutable KTwoVector fOrigin;
    mutable KTwoVector fXUnit;
    mutable KTwoVector fYUnit;

    void Initialize() const;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif

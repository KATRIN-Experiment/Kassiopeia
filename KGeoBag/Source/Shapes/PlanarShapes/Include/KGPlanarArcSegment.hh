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
    KGPlanarArcSegment(const katrin::KTwoVector& aStart, const katrin::KTwoVector& anEnd, const double& aRadius, const bool& isRight,
                       const bool& isShort, const unsigned int aCount = 16);
    KGPlanarArcSegment(const double& anX1, const double& aY1, const double& anX2, const double& aY2,
                       const double& aRadius, const bool& isRight, const bool& isShort, const unsigned int aCount = 16);
    ~KGPlanarArcSegment() override;

    static std::string Name()
    {
        return "arc_segment";
    }

    KGPlanarArcSegment* Clone() const override;
    void CopyFrom(const KGPlanarArcSegment& aCopy);

  public:
    void Start(const katrin::KTwoVector& aStart);
    void X1(const double& aValue);
    void Y1(const double& aValue);
    void End(const katrin::KTwoVector& anEnd);
    void X2(const double& aValue);
    void Y2(const double& aValue);
    void Radius(const double& aValue);
    void Right(const bool& aValue);
    void Short(const bool& aValue);
    void MeshCount(const unsigned int& aCount);

    const katrin::KTwoVector& Start() const override;
    const double& X1() const;
    const double& Y1() const;
    const katrin::KTwoVector& End() const override;
    const double& X2() const;
    const double& Y2() const;
    const double& Radius() const;
    const bool& Right() const;
    const bool& Short() const;
    const unsigned int& MeshCount() const;

    const double& Length() const override;
    const double& Angle() const;
    const katrin::KTwoVector& Centroid() const override;
    const katrin::KTwoVector& Origin() const;
    const katrin::KTwoVector& XUnit() const;
    const katrin::KTwoVector& YUnit() const;

  public:
    katrin::KTwoVector At(const double& aLength) const override;
    katrin::KTwoVector Point(const katrin::KTwoVector& aQuery) const override;
    katrin::KTwoVector Normal(const katrin::KTwoVector& aQuery) const override;
    bool Above(const katrin::KTwoVector& aQuery) const override;

  private:
    katrin::KTwoVector fStart;
    katrin::KTwoVector fEnd;
    double fRadius;
    bool fRight;
    bool fShort;
    unsigned int fMeshCount;

    mutable double fLength;
    mutable double fAngle;
    mutable katrin::KTwoVector fCentroid;
    mutable katrin::KTwoVector fOrigin;
    mutable katrin::KTwoVector fXUnit;
    mutable katrin::KTwoVector fYUnit;

    void Initialize() const;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif

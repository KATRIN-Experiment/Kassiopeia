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
    KGPlanarLineSegment(const katrin::KTwoVector& aStart, const katrin::KTwoVector& anEnd, const unsigned int aCount = 2,
                        const double aPower = 1.);
    KGPlanarLineSegment(const double& anX1, const double& aY1, const double& anX2, const double& aY2,
                        const unsigned int aCount = 2, const double aPower = 1.);
    ~KGPlanarLineSegment() override;

    static std::string Name()
    {
        return "line_segment";
    }

    KGPlanarLineSegment* Clone() const override;
    void CopyFrom(const KGPlanarLineSegment& aCopy);

  public:
    void Start(const katrin::KTwoVector& aStart);
    void X1(const double& aValue);
    void Y1(const double& aValue);
    void End(const katrin::KTwoVector& anEnd);
    void X2(const double& aValue);
    void Y2(const double& aValue);
    void MeshCount(const unsigned int& aCount);
    void MeshPower(const double& aPower);

    const katrin::KTwoVector& Start() const override;
    const double& X1() const;
    const double& Y1() const;
    const katrin::KTwoVector& End() const override;
    const double& X2() const;
    const double& Y2() const;
    const unsigned int& MeshCount() const;
    const double& MeshPower() const;

    const double& Length() const override;
    const katrin::KTwoVector& Centroid() const override;
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
    unsigned int fMeshCount;
    double fMeshPower;

    mutable double fLength;
    mutable katrin::KTwoVector fCentroid;
    mutable katrin::KTwoVector fXUnit;
    mutable katrin::KTwoVector fYUnit;

    void Initialize() const;
    mutable bool fInitialized;
};

}  // namespace KGeoBag

#endif

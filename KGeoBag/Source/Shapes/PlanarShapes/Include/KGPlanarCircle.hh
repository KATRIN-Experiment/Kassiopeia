#ifndef KGPLANARCIRCLE_HH_
#define KGPLANARCIRCLE_HH_

#include "KGPlanarClosedPath.hh"

namespace KGeoBag
{

class KGPlanarCircle : public KGPlanarClosedPath
{
  public:
    KGPlanarCircle();
    KGPlanarCircle(const KGPlanarCircle& aCopy);
    KGPlanarCircle(const KTwoVector& aCentroid, const double& aRadius, const unsigned int aCount = 32);
    KGPlanarCircle(const double& anX, const double& aY, const double& aRadius, const unsigned int aCount = 32);
    ~KGPlanarCircle() override;

    KGPlanarCircle* Clone() const override;
    void CopyFrom(const KGPlanarCircle& aCopy);

  public:
    void Centroid(const KTwoVector& aStart);
    void X(const double& aValue);
    void Y(const double& aValue);
    void Radius(const double& aValue);
    void MeshCount(const unsigned int& aCount);

    const KTwoVector& Centroid() const override;
    const double& X() const;
    const double& Y() const;
    const double& Radius() const;
    const unsigned int& MeshCount() const;

    const double& Length() const override;
    const KTwoVector& Anchor() const override;

  public:
    KTwoVector At(const double& aLength) const override;
    KTwoVector Point(const KTwoVector& aQuery) const override;
    KTwoVector Normal(const KTwoVector& aQuery) const override;
    bool Above(const KTwoVector& aQuery) const override;

  private:
    KTwoVector fCentroid;
    double fRadius;
    unsigned int fMeshCount;

    mutable double fLength;
    mutable KTwoVector fAnchor;

    void Initialize() const;
    mutable bool fInitialized;
};
}  // namespace KGeoBag

#endif

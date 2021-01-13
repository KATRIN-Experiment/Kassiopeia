#ifndef KTRANSFORMATION_H_
#define KTRANSFORMATION_H_

#include "KRotation.hh"
#include "KThreeVector.hh"

namespace KGeoBag
{

class KTransformation
{
  public:
    KTransformation();
    KTransformation(const KTransformation& aTransformation);
    virtual ~KTransformation();

    void Apply(KThreeVector& point) const;
    void ApplyRotation(KThreeVector& point) const;
    void ApplyDisplacement(KThreeVector& point) const;

    void ApplyInverse(KThreeVector& point) const;
    void ApplyRotationInverse(KThreeVector& point) const;
    void ApplyDisplacementInverse(KThreeVector& point) const;

    //*****************
    //coordinate system
    //*****************

  public:
    void SetOrigin(const KGeoBag::KThreeVector& origin);
    void SetFrameAxisAngle(const double& angle, const double& theta, const double& phi);
    void SetFrameEuler(const double& alpha, const double& beta, const double& gamma);
    void SetXAxis(const KGeoBag::KThreeVector& xaxis);
    void SetYAxis(const KGeoBag::KThreeVector& yaxis);
    void SetZAxis(const KGeoBag::KThreeVector& zaxis);
    const KGeoBag::KThreeVector& GetOrigin() const;
    const KGeoBag::KThreeVector& GetXAxis() const;
    const KGeoBag::KThreeVector& GetYAxis() const;
    const KGeoBag::KThreeVector& GetZAxis() const;

  private:
    void LocalFromGlobal(const KGeoBag::KThreeVector& point, KGeoBag::KThreeVector& target) const;
    void GlobalFromLocal(const KGeoBag::KThreeVector& point, KGeoBag::KThreeVector& target) const;

    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;

    //********
    //rotation
    //********

  public:
    void SetRotationAxisAngle(const double& angle, const double& theta, const double& phi);
    void SetRotationEuler(const double& phi, const double& theta, const double& psi);
    void SetRotationZYZEuler(const double& phi, const double& theta, const double& psi);
    void SetRotatedFrame(const KGeoBag::KThreeVector& x, const KGeoBag::KThreeVector& y,
                         const KGeoBag::KThreeVector& z);
    const KRotation& GetRotation() const;

  private:
    KRotation fRotation;
    KRotation fRotationInverse;

    //************
    //displacement
    //************

  public:
    void SetDisplacement(const double& xdisp, const double& ydisp, const double& zdisp);
    void SetDisplacement(const KGeoBag::KThreeVector& disp);
    const KGeoBag::KThreeVector& GetDisplacement() const;

  private:
    KGeoBag::KThreeVector fDisplacement;
};

inline const KGeoBag::KThreeVector& KTransformation::GetOrigin() const
{
    return fOrigin;
}
inline const KGeoBag::KThreeVector& KTransformation::GetXAxis() const
{
    return fXAxis;
}
inline const KGeoBag::KThreeVector& KTransformation::GetYAxis() const
{
    return fYAxis;
}
inline const KGeoBag::KThreeVector& KTransformation::GetZAxis() const
{
    return fZAxis;
}

inline const KGeoBag::KThreeVector& KTransformation::GetDisplacement() const
{
    return fDisplacement;
}

inline const KRotation& KTransformation::GetRotation() const
{
    return fRotation;
}

}  // namespace KGeoBag

#endif

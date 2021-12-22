#ifndef KROTATION_H_
#define KROTATION_H_

#include "KThreeMatrix.hh"

#include <vector>

namespace katrin
{

class KRotation : public KThreeMatrix
{
  public:
    KRotation();
    KRotation(const KRotation& aRotation);
    KRotation& operator=(const KRotation& other) = default;
    ~KRotation() override;

    KRotation& operator=(const KThreeMatrix& aMatrix);

    void SetIdentity();
    void SetAxisAngle(const KThreeVector& anAxis, const double& anAngle);
    void SetAxisAngleInDegrees(const KThreeVector& anAxis, const double& anAngle);
    void SetEulerAngles(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerAnglesInDegrees(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerAngles(const std::vector<double>& anArray);
    void SetEulerAnglesInDegrees(const std::vector<double>& anArray);
    void SetEulerZYZAngles(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerZYZAnglesInDegrees(const double& anAlpha, const double& aBeta, const double& aGamma);
    void SetEulerZYZAngles(const std::vector<double>& anArray);
    void SetEulerZYZAnglesInDegrees(const std::vector<double>& anArray);
    void SetRotatedFrame(const KThreeVector& x, const KThreeVector& y,
                         const KThreeVector& z);

    void GetEulerAngles(double& anAlpha, double& aBeta, double& aGamma) const;
    void GetEulerAnglesInDegrees(double& anAlpha, double& aBeta, double& aGamma) const;
};

}  // namespace katrin

#endif

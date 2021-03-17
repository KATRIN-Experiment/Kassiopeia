#include "KTransformation.hh"

#include "KConst.h"
#include "KGMathMessage.hh"

#include <cmath>
#include <iostream>

namespace KGeoBag
{

KTransformation::KTransformation() :
    fOrigin(0., 0., 0.),
    fXAxis(1., 0., 0.),
    fYAxis(0., 1., 0.),
    fZAxis(0., 0., 1.),
    fDisplacement(0., 0., 0.)
{}
KTransformation::KTransformation(const KTransformation&) = default;
KTransformation::~KTransformation() = default;

void KTransformation::Apply(KThreeVector& point) const
{
    KThreeVector LocalPoint;

    LocalFromGlobal(point, LocalPoint);
    LocalPoint = fRotation * LocalPoint;
    LocalPoint = fDisplacement + LocalPoint;
    GlobalFromLocal(LocalPoint, point);

    return;
}
void KTransformation::ApplyRotation(KThreeVector& point) const
{
    KThreeVector LocalPoint;

    LocalFromGlobal(point, LocalPoint);
    LocalPoint = fRotation * LocalPoint;
    GlobalFromLocal(LocalPoint, point);

    return;
}
void KTransformation::ApplyDisplacement(KThreeVector& point) const
{
    KThreeVector LocalPoint;

    LocalFromGlobal(point, LocalPoint);
    LocalPoint = fDisplacement + LocalPoint;
    GlobalFromLocal(LocalPoint, point);

    return;
}

void KTransformation::ApplyInverse(KThreeVector& point) const
{
    KThreeVector LocalPoint;

    LocalFromGlobal(point, LocalPoint);
    LocalPoint = LocalPoint - fDisplacement;
    LocalPoint = fRotationInverse * LocalPoint;
    GlobalFromLocal(LocalPoint, point);

    return;
}
void KTransformation::ApplyRotationInverse(KThreeVector& point) const
{
    KThreeVector LocalPoint;

    LocalFromGlobal(point, LocalPoint);
    LocalPoint = fRotationInverse * LocalPoint;
    GlobalFromLocal(LocalPoint, point);

    return;
}
void KTransformation::ApplyDisplacementInverse(KThreeVector& point) const
{
    KThreeVector LocalPoint;

    LocalFromGlobal(point, LocalPoint);
    LocalPoint = LocalPoint - fDisplacement;
    GlobalFromLocal(LocalPoint, point);

    return;
}

void KTransformation::SetOrigin(const KThreeVector& origin)
{
    fOrigin = origin;

    return;
}
void KTransformation::SetFrameAxisAngle(const double& angle, const double& theta, const double& phi)
{
    double CosTheta = cos((katrin::KConst::Pi() / 180.) * theta);
    double SinTheta = sin((katrin::KConst::Pi() / 180.) * theta);
    double CosPhi = cos((katrin::KConst::Pi() / 180.) * phi);
    double SinPhi = sin((katrin::KConst::Pi() / 180.) * phi);

    KThreeVector Axis(SinTheta * CosPhi, SinTheta * SinPhi, CosTheta);
    double Angle = (katrin::KConst::Pi() / 180.) * angle;

    KRotation Orientation;
    Orientation.SetAxisAngle(Axis, Angle);

    fXAxis[0] = 1.;
    fXAxis[1] = 0.;
    fXAxis[2] = 0.;
    fXAxis = Orientation * fXAxis;

    fYAxis[0] = 0.;
    fYAxis[1] = 1.;
    fYAxis[2] = 0.;
    fYAxis = Orientation * fYAxis;

    fZAxis[0] = 0.;
    fZAxis[1] = 0.;
    fZAxis[2] = 1.;
    fZAxis = Orientation * fZAxis;

    return;
}
void KTransformation::SetFrameEuler(const double& phi, const double& theta, const double& psi)
{
    double Phi = (katrin::KConst::Pi() / 180.) * phi;
    double Theta = (katrin::KConst::Pi() / 180.) * theta;
    double Psi = (katrin::KConst::Pi() / 180.) * psi;

    KRotation Orientation;
    Orientation.SetEulerAngles(Phi, Theta, Psi);

    fXAxis[0] = 1.;
    fXAxis[1] = 0.;
    fXAxis[2] = 0.;
    fXAxis = Orientation * fXAxis;

    fYAxis[0] = 0.;
    fYAxis[1] = 1.;
    fYAxis[2] = 0.;
    fYAxis = Orientation * fYAxis;

    fZAxis[0] = 0.;
    fZAxis[1] = 0.;
    fZAxis[2] = 1.;
    fZAxis = Orientation * fZAxis;

    return;
}
void KTransformation::SetXAxis(const KThreeVector& localx)
{
    fXAxis = localx;
    return;
}
void KTransformation::SetYAxis(const KThreeVector& localy)
{
    fYAxis = localy;
    return;
}
void KTransformation::SetZAxis(const KThreeVector& localz)
{
    fZAxis = localz;
    return;
}
void KTransformation::LocalFromGlobal(const KThreeVector& point, KThreeVector& target) const
{
    KThreeVector PSub = point - fOrigin;

    target[0] = PSub.Dot(fXAxis);
    target[1] = PSub.Dot(fYAxis);
    target[2] = PSub.Dot(fZAxis);

    mathmsg_debug("converting global point " << point << " to local point " << target << eom;)

        return;
}
void KTransformation::GlobalFromLocal(const KThreeVector& point, KThreeVector& target) const
{
    KThreeVector PSub = point[0] * fXAxis + point[1] * fYAxis + point[2] * fZAxis;

    target = fOrigin + PSub;

    mathmsg_debug("converting local point " << point << " to global point " << target << eom;)


        return;
}

void KTransformation::SetDisplacement(const double& xdisp, const double& ydisp, const double& zdisp)
{
    fDisplacement[0] = xdisp;
    fDisplacement[1] = ydisp;
    fDisplacement[2] = zdisp;
    return;
}

void KTransformation::SetDisplacement(const KThreeVector& disp)
{
    fDisplacement = disp;
    return;
}

void KTransformation::SetRotationAxisAngle(const double& anAngle, const double& aTheta, const double& aPhi)
{
    double tCosTheta = cos((katrin::KConst::Pi() / 180.) * aTheta);
    double tSinTheta = sin((katrin::KConst::Pi() / 180.) * aTheta);
    double tCosPhi = cos((katrin::KConst::Pi() / 180.) * aPhi);
    double tSinPhi = sin((katrin::KConst::Pi() / 180.) * aPhi);

    KThreeVector tAxis(tSinTheta * tCosPhi, tSinTheta * tSinPhi, tCosTheta);
    double tRadianAngle = (katrin::KConst::Pi() / 180.) * anAngle;

    fRotation.SetAxisAngle(tAxis, tRadianAngle);
    fRotationInverse = fRotation.Inverse();

    return;
}

void KTransformation::SetRotationEuler(const double& anAlpha, const double& aBeta, const double& aGamma)
{
    double tRadianAlpha = (katrin::KConst::Pi() / 180.) * anAlpha;
    double tRadianBeta = (katrin::KConst::Pi() / 180.) * aBeta;
    double tRadianGamma = (katrin::KConst::Pi() / 180.) * aGamma;

    fRotation.SetEulerAngles(tRadianAlpha, tRadianBeta, tRadianGamma);
    fRotationInverse = fRotation.Inverse();

    return;
}

void KTransformation::SetRotationZYZEuler(const double& anAlpha, const double& aBeta, const double& aGamma)
{
    double tRadianAlpha = (katrin::KConst::Pi() / 180.) * anAlpha;
    double tRadianBeta = (katrin::KConst::Pi() / 180.) * aBeta;
    double tRadianGamma = (katrin::KConst::Pi() / 180.) * aGamma;

    fRotation.SetEulerZYZAngles(tRadianAlpha, tRadianBeta, tRadianGamma);
    fRotationInverse = fRotation.Inverse();

    return;
}

void KTransformation::SetRotatedFrame(const KThreeVector& x, const KThreeVector& y, const KThreeVector& z)
{
    fRotation.SetRotatedFrame(x, y, z);
    fRotationInverse = fRotation.Inverse();

    return;
}

}  // namespace KGeoBag

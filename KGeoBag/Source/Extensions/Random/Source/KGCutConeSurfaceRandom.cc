/*
 * KGCutConeSurfaceRandom.cc
 *
 *  Created on: 20.05.2014
 *      Author: user
 */

#include "KGCutConeSurfaceRandom.hh"

void KGeoBag::KGCutConeSurfaceRandom::VisitCutConeSurface(KGeoBag::KGCutConeSurface* aCutConeSurface)
{
    katrin::KThreeVector point;

    double r1 = aCutConeSurface->R1();
    double r2 = aCutConeSurface->R2();
    double z1 = aCutConeSurface->Z1();
    double z2 = aCutConeSurface->Z2();

    double SmallRadius, SmallRadiusPosition, LargeRadius;
    if (r1 < r2) {
        SmallRadius = r1;
        SmallRadiusPosition = z1;
        LargeRadius = r2;
    }
    else {
        SmallRadius = r2;
        SmallRadiusPosition = z2;
        LargeRadius = r1;
    }

    double ConeLength = z2 - z1;
    double ConeSpread = r2 - r1;
    double ConeSlope = ConeSpread / ConeLength;
    double SmallZeta = SmallRadius * ConeLength / ConeSpread;
    double LargeZeta = LargeRadius * ConeLength / ConeSpread;
    double Ratio = (SmallRadius * SmallRadius) / (LargeRadius * LargeRadius);
    double ZPosition = sqrt(Ratio + (1 - Ratio) * Uniform()) * LargeZeta - SmallZeta;
    double r = SmallRadius + ConeSlope * ZPosition;
    double phi = Uniform() * 2 * katrin::KConst::Pi();

    point.SetZ(SmallRadiusPosition + ZPosition);
    point.SetX(cos(phi) * r);
    point.SetY(sin(phi) * r);

    SetRandomPoint(point);
}

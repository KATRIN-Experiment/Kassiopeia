#include "KFMTrianglePolarBasisCalculator.hh"


namespace KEMField
{


KFMTrianglePolarBasisCalculator::KFMTrianglePolarBasisCalculator()
{
    ;
}


KFMTrianglePolarBasisCalculator::~KFMTrianglePolarBasisCalculator()
{
    ;
}


void KFMTrianglePolarBasisCalculator::Convert(const KFMPointCloud<3>* vertices, KFMTrianglePolarBasis& basis)
{
    SetPointCloud(vertices);
    ConstructBasis();
    basis.h = fH;
    basis.area = fArea;
    basis.phi1 = fPhi1;
    basis.phi2 = fPhi2;

    //components of the x-axis unit vector
    basis.e0x = fX[0];
    basis.e0y = fX[1];
    basis.e0z = fX[2];

    //componets of the y-axis unit vector
    basis.e1x = fY[0];
    basis.e1y = fY[1];
    basis.e1z = fY[2];

    //components of the z-axis unit vector
    basis.e2x = fZ[0];
    basis.e2y = fZ[1];
    basis.e2z = fZ[2];
}


void KFMTrianglePolarBasisCalculator::SetPointCloud(const KFMPointCloud<3>* vertices)
{
    for (unsigned int i = 0; i < 3; i++) {
        for (unsigned int j = 0; j < 3; j++) {
            fP[i][j] = vertices->GetPoint(i)[j];
        }
    }
}


void KFMTrianglePolarBasisCalculator::ConstructBasis()
{
    //vectors along triangle sides
    Subtract(fP[1], fP[0], fN0_1);
    Subtract(fP[2], fP[0], fN0_2);
    Cross(fN0_1, fN0_2, fNPerp);
    fArea = 0.5 * Magnitude(fNPerp);
    Normalize(fNPerp);

    //have to construct the x, y and z-axes
    double v[3];
    Subtract(fP[2], fP[1], v);

    //the line pointing along v is the y-axis
    SetEqual(v, fY);
    Normalize(fY);

    //q is closest point to fP[0] on line connecting fP[1] to fP[2]
    double t = (Dot(fP[0], v) - Dot(fP[1], v)) / (Dot(v, v));
    ScalarMultiply(t, v);
    Add(fP[1], v, fQ);

    //the line going from fP[0] to fQ is the x-axis
    Subtract(fQ, fP[0], fX);
    //gram-schmidt out any y-axis component in the x-axis
    double proj = Dot(fX, fY);
    double sub[3];
    SetEqual(fY, sub);
    ScalarMultiply(proj, sub);
    Subtract(fX, sub, fX);
    fH = Magnitude(fX);  //compute triangle height along x
    Normalize(fX);


    //form the z-axis by the cross product
    Cross(fX, fY, fZ);
    Normalize(fZ);

    //now we need to find the angles from the x-axis
    double temp[3];
    Subtract(fP[1], fP[0], temp);
    double PAy = Dot(temp, fY);

    Subtract(fP[1], fP[0], temp);
    double PAx = Dot(temp, fX);

    Subtract(fP[2], fP[0], temp);
    double PBy = Dot(temp, fY);

    Subtract(fP[2], fP[0], temp);
    double PBx = Dot(temp, fX);

    fPhi1 = std::atan2(PAy, PAx);
    fPhi2 = std::atan2(PBy, PBx);
}


}  // namespace KEMField

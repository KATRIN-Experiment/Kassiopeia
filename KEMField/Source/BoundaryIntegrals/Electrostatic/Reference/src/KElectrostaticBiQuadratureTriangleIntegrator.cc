#include "KElectrostaticBiQuadratureTriangleIntegrator.hh"

namespace KEMField
{

KThreeVector triCommonFieldPoint;  // = Pcommon

KThreeVector triCommonP1;  // Bcommon

KThreeVector triCommonAlongSideP2P0;      // = acommon
KThreeVector triCommonAlongSideP2P1;      // = bcommon
KThreeVector triCommonAlongSideP2P0Unit;  // = uacommon

double triCommonHeightP2onP0P1;       // = hcommon
KThreeVector triCommonAlongSideP0P1;  // = ccommon

KThreeVector triCommonHeightIterSideP2P1;  // = Ecommon

unsigned short triComputationFlag;  // 0: Ex,  1:  Ey,  2:  Ez,  3:  Phi

// integration number for 1-dimensional Gauss-Legendre  quadrature
unsigned int triIntNodes = 32;


double KElectrostaticBiQuadratureTriangleIntegrator::triF1(double y)
{
    KThreeVector triHeightIterSideP2P1;  // = E
    KThreeVector triHeightIterSideP0P1;  // = F
    double triLengthConnectSideIter;     // = L

    triCommonHeightIterSideP2P1 = triCommonP1 - ((y / triCommonHeightP2onP0P1) * triCommonAlongSideP2P1);
    triHeightIterSideP2P1 = triCommonHeightIterSideP2P1;
    triHeightIterSideP0P1 = triCommonP1 - ((y / triCommonHeightP2onP0P1) * triCommonAlongSideP0P1);

    triLengthConnectSideIter = (triHeightIterSideP2P1 - triHeightIterSideP0P1).Magnitude();

    return triQuadGaussLegendreVarN(&KElectrostaticBiQuadratureTriangleIntegrator::triF2,
                                    0.,
                                    triLengthConnectSideIter,
                                    triIntNodes);
}

double KElectrostaticBiQuadratureTriangleIntegrator::triF2(double x)
{
    KThreeVector Q = triCommonHeightIterSideP2P1 + x * triCommonAlongSideP2P0Unit;
    return KElectrostaticBiQuadratureTriangleIntegrator::triF(Q, triCommonFieldPoint);
}

double KElectrostaticBiQuadratureTriangleIntegrator::triF(KThreeVector Q, KThreeVector P)
{
    double R, C3, C;
    KThreeVector QP;
    QP = P - Q;
    R = QP.Magnitude();
    C = 1. / R;
    C3 = C * C * C;

    if (triComputationFlag == 3)
        return KEMConstants::OneOverFourPiEps0 * C;

    for (unsigned short j = 0; j < 3; j++) {
        if (triComputationFlag == j)
            return KEMConstants::OneOverFourPiEps0 * C3 * QP[j];
    }

    return 0.;
}

double KElectrostaticBiQuadratureTriangleIntegrator::triQuadGaussLegendreVarN(double (*f)(double), double a, double b,
                                                                              unsigned int n)
{
    KGaussLegendreQuadrature fIntegrator;
    double Integral, xmin, xmax, del, ret;
    if (n <= 32)
        fIntegrator(f, a, b, n, &Integral);
    else {
        unsigned int imax = n / 32 + 1;
        Integral = 0.;
        del = (b - a) / imax;
        for (unsigned int i = 1; i <= imax; i++) {
            xmin = a + del * (i - 1);
            xmax = xmin + del;
            fIntegrator(f, xmin, xmax, 32, &ret);
            Integral += ret;
        }
    }
    return Integral;
}

double KElectrostaticBiQuadratureTriangleIntegrator::Potential(const KTriangle* source, const KPosition& P) const
{
    triCommonFieldPoint = P;
    double phi = 0.;

    triComputationFlag = 3;

    // corner points of the triangle
    const KThreeVector triP0 = source->GetP0();  // = A

    const KThreeVector triP1 = source->GetP1();  // = B
    triCommonP1 = triP1;                         // = Bcommon

    const KThreeVector triP2 = source->GetP2();  // = C

    const KThreeVector triAlongSideP2P0 = triP0 - triP2;  // = a
    triCommonAlongSideP2P0 = triAlongSideP2P0;            // set global value

    const KThreeVector triAlongSideP2P1 = triP1 - triP2;  // = b
    triCommonAlongSideP2P1 = triAlongSideP2P1;            // set global value

    const double triAlongSideP2P0Length = triAlongSideP2P0.Magnitude();  // = absa
    const double triAlongSideP2P1Length = triAlongSideP2P1.Magnitude();  // = absb

    triCommonAlongSideP2P0Unit = (1. / triAlongSideP2P0Length) * triAlongSideP2P0;  // = uacommon

    const double cosgamma = triAlongSideP2P0.Dot(triAlongSideP2P1) / (triAlongSideP2P0Length * triAlongSideP2P1Length);
    const KThreeVector triProjP2onP0P1 =
        triP2 + (triAlongSideP2P1Length * cosgamma) * triCommonAlongSideP2P0Unit;  // = D

    const double triHeightP2onP0P1 = (triProjP2onP0P1 - triP1).Magnitude();  // = h
    triCommonHeightP2onP0P1 = triHeightP2onP0P1;                             // = hcommon

    const KThreeVector triAlongSideP0P1 = triP1 - triP0;  // = c
    triCommonAlongSideP0P1 = triAlongSideP0P1;            // = ccommon

    phi = triQuadGaussLegendreVarN(&KElectrostaticBiQuadratureTriangleIntegrator::triF1,
                                   0.,
                                   triHeightP2onP0P1,
                                   triIntNodes);

    return phi;
}

KThreeVector KElectrostaticBiQuadratureTriangleIntegrator::ElectricField(const KTriangle* source,
                                                                         const KPosition& P) const
{
    triCommonFieldPoint = P;
    KThreeVector eField(0., 0., 0.);

    // corner points of the triangle
    const KThreeVector triP0 = source->GetP0();  // A
    const KThreeVector triP1 = source->GetP1();  // B
    const KThreeVector triP2 = source->GetP2();  // C

    KThreeVector triAlongSideP2P0 = triP0 - triP2;  // = a
    triCommonAlongSideP2P0 = triAlongSideP2P0;      // set global value

    KThreeVector triAlongSideP2P1 = triP1 - triP2;  // = b
    triCommonAlongSideP2P1 = triAlongSideP2P1;      // set global value

    const double triAlongSideP2P0Length = triAlongSideP2P0.Magnitude();  // = absa
    const double triAlongSideP2P1Length = triAlongSideP2P1.Magnitude();  // = absb

    triCommonAlongSideP2P0Unit = (1 / triAlongSideP2P0Length) * triAlongSideP2P0;  // = uacommon

    const double cosgamma = triAlongSideP2P0.Dot(triAlongSideP2P1) / (triAlongSideP2P0Length * triAlongSideP2P1Length);
    const KThreeVector triProjP2onP0P1 =
        triP2 + (triAlongSideP2P1Length * cosgamma) * triCommonAlongSideP2P0Unit;  // = D

    const double triHeightP2onP0P1 = (triProjP2onP0P1 - triP1).Magnitude();  // = h
    triCommonHeightP2onP0P1 = triHeightP2onP0P1;                             // = hcommon

    KThreeVector triAlongSideP0P1 = triP1 - triP0;  // = c
    triCommonAlongSideP0P1 = triAlongSideP0P1;      // = ccommon

    for (unsigned short i = 0; i < 3; i++) {
        triComputationFlag = i;
        eField[i] = triQuadGaussLegendreVarN(&KElectrostaticBiQuadratureTriangleIntegrator::triF1,
                                             0.,
                                             triHeightP2onP0P1,
                                             triIntNodes);
    }
    return eField;
}

std::pair<KThreeVector, double>
KElectrostaticBiQuadratureTriangleIntegrator::ElectricFieldAndPotential(const KTriangle* source,
                                                                        const KPosition& P) const
{
    return std::make_pair(ElectricField(source, P), Potential(source, P));
}

double KElectrostaticBiQuadratureTriangleIntegrator::Potential(const KSymmetryGroup<KTriangle>* source,
                                                               const KPosition& P) const
{
    double potential = 0.;
    for (auto it = source->begin(); it != source->end(); ++it)
        potential += Potential(*it, P);
    return potential;
}

KThreeVector KElectrostaticBiQuadratureTriangleIntegrator::ElectricField(const KSymmetryGroup<KTriangle>* source,
                                                                         const KPosition& P) const
{
    KThreeVector electricField(0., 0., 0.);
    for (auto it = source->begin(); it != source->end(); ++it)
        electricField += ElectricField(*it, P);
    return electricField;
}

std::pair<KThreeVector, double>
KElectrostaticBiQuadratureTriangleIntegrator::ElectricFieldAndPotential(const KSymmetryGroup<KTriangle>* source,
                                                                        const KPosition& P) const
{
    std::pair<KThreeVector, double> fieldAndPotential;
    double potential(0.);
    KThreeVector electricField(0., 0., 0.);

    for (auto it = source->begin(); it != source->end(); ++it) {
        fieldAndPotential = ElectricFieldAndPotential(*it, P);
        electricField += fieldAndPotential.first;
        potential += fieldAndPotential.second;
    }

    return std::make_pair(electricField, potential);
}

}  // namespace KEMField

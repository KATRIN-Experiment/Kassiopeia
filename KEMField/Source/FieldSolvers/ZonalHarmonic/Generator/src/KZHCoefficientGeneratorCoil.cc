#include "KZHCoefficientGeneratorCoil.hh"

#include "KEMConstants.hh"

namespace KEMField
{
/**
   * Adds the contribution made due the coil to the central coefficents coeff.
   */
void KZHCoefficientGenerator<KCoil>::ComputeCentralCoefficients(double z0, double rho, std::vector<double>& coeff) const
{
    int nCoeffs = coeff.size();

    std::vector<double> P1p(nCoeffs + 2);
    std::vector<double> b0(nCoeffs + 1);
    std::vector<double> b1(nCoeffs + 1);
    std::vector<std::vector<double>> bhat(nCoeffs + 1, std::vector<double>(31, 0));

    static int M = 30;
    // slightly modified Newton-Cotes coefficients
    static double w9[10] = {0.2803440531305107e0,
                            0.1648702325837748e1,
                            -0.2027449845679092e0,
                            0.2797927414021179e1,
                            -0.9761199294532843e0,
                            0.2556499393738999e1,
                            0.1451083002645404e0,
                            0.1311227127425048e1,
                            0.9324249063051143e0,
                            0.1006631393298060e1};
    static double w[31];

    // Initialization of the integration weight factors (set once during the
    // calculation of the first source point)
    int m;
    for (m = 0; m <= 9; m++)
        w[m] = w9[m];
    for (m = 10; m <= M - 10; m++)
        w[m] = 1.;
    for (m = M - 9; m <= M; m++)
        w[m] = w9[M - m];

    // source point calculation loop
    double Z, R, q, z, u, constcen, rcen, rcen1, ro;
    double del = (fCoil->GetR1() - fCoil->GetR0()) / M;
    double sigmac = Prefactor() / ((fCoil->GetZ1() - fCoil->GetZ0()) * (fCoil->GetR1() - fCoil->GetR0()));

    // integration loop

    for (int m = 0; m <= M; m++) {
        R = fCoil->GetR0() + del * m;
        for (int iz = 0; iz < 2; iz++) {
            if (iz == 0)
                Z = fCoil->GetZ1();
            else
                Z = fCoil->GetZ0();
            z = Z - z0;
            ro = sqrt(R * R + z * z);
            u = z / ro;

            // fill the Legendre polynomial array

            P1p[0] = 0.;
            P1p[1] = 1.;

            for (int n = 2; n < nCoeffs + 2; n++)
                P1p[n] = ((2 * n - 1) * u * P1p.at(n - 1) - n * P1p.at(n - 2)) / (1. * (n - 1.));

            constcen = KEMConstants::Mu0 * sigmac / 2. * (1. - u * u) / rho;
            rcen = rho / ro;
            rcen1 = rcen;
            for (int n = 0; n < nCoeffs + 1; n++) {
                if (iz == 0)
                    b0[n] = constcen * rcen1 * P1p[n + 1];
                else if (iz == 1)
                    b1[n] = constcen * rcen1 * P1p[n + 1];
                rcen1 *= rcen;
            }
        }

        Z = fCoil->GetZ1();
        z = Z - z0;
        ro = sqrt(R * R + z * z);
        bhat[0][m] = KEMConstants::Mu0 * sigmac / 2. * z / ro;
        Z = fCoil->GetZ0();
        z = Z - z0;
        ro = sqrt(R * R + z * z);
        bhat[0][m] -= KEMConstants::Mu0 * sigmac / 2. * z / ro;
        for (int n = 1; n < nCoeffs + 1; n++) {
            bhat[n][m] = -rho / n * (b0[n - 1] - b1[n - 1]);
        }
    }

    for (int n = 0; n < nCoeffs; n++) {
        q = 0;
        for (int m = 0; m <= M; m++)
            q += bhat[n][m] * w[m];
        q *= del;
        coeff[n] += q;
    }
}

/**
   * Adds the contribution made due the coil to the remote coefficents <coeff>.
   */
void KZHCoefficientGenerator<KCoil>::ComputeRemoteCoefficients(double z0, double rho, std::vector<double>& coeff) const
{
    int nCoeffs = coeff.size();

    std::vector<double> Pp(nCoeffs + 2);
    std::vector<std::vector<double>> bs(nCoeffs + 2, std::vector<double>(2, 0));
    std::vector<std::vector<double>> bshat(nCoeffs + 1, std::vector<double>(1001, 0));

    // slightly modified Newton-Cotes coefficients
    static double w9[10] = {0.2803440531305107e0,
                            0.1648702325837748e1,
                            -0.2027449845679092e0,
                            0.2797927414021179e1,
                            -0.9761199294532843e0,
                            0.2556499393738999e1,
                            0.1451083002645404e0,
                            0.1311227127425048e1,
                            0.9324249063051143e0,
                            0.1006631393298060e1};
    double w[1001];  // integration weight factors

    // radial integration number M:
    double ratio = (fCoil->GetR1() - fCoil->GetR0()) / fCoil->GetR0();
    int M;
    if (ratio < 0.1)
        M = 30;
    else if (ratio >= 0.1 && ratio < 0.2)
        M = 50;
    else
        M = 60 * ratio / 0.2;
    if (M > 1000)
        M = 1000;

    // initialization of the integration weight factors:
    for (int m = 0; m <= 9; m++)
        w[m] = w9[m];
    for (int m = 10; m <= M - 10; m++)
        w[m] = 1.;
    for (int m = M - 9; m <= M; m++)
        w[m] = w9[M - m];

    // initialization of Pp[n]:
    Pp[0] = 0.;
    Pp[1] = 1.;

    // calculation of coil i's contribution to source point coeffs coeffs[]
    double del = (fCoil->GetR1() - fCoil->GetR0()) / M;
    double sigmac = Prefactor() / (fCoil->GetZ1() - fCoil->GetZ0()) / (fCoil->GetR1() - fCoil->GetR0());

    // integration loop
    double R, Z, z, ro, u, constrem, rrem, rrem1;
    for (int m = 0; m <= M; m++) {
        R = fCoil->GetR0() + del * m;
        for (int iz = 0; iz < 2; iz++) {
            if (iz == 0)
                Z = fCoil->GetZ1();
            else
                Z = fCoil->GetZ0();

            z = Z - z0;
            ro = sqrt(R * R + z * z);
            u = z / ro;

            for (int n = 2; n < nCoeffs + 2; n++)
                Pp[n] = ((2 * n - 1) * u * Pp[n - 1] - n * Pp[n - 2]) / (1. * (n - 1.));

            constrem = KEMConstants::Mu0 * sigmac / 2. * (1. - u * u) / rho;
            rrem = ro / rho;
            rrem1 = rrem * rrem;

            for (int n = 2; n < nCoeffs + 2; n++) {
                bs[n][iz] = constrem * rrem1 * Pp[n - 1];
                rrem1 *= rrem;
            }
        }

        bshat[0][m] = bshat[1][m] = 0.;

        for (int n = 2; n <= nCoeffs; n++)
            bshat[n][m] = rho / (n + 1.) * (bs[n + 1][0] - bs[n + 1][1]);
    }
    // end of m-loop

    for (int n = 0; n < nCoeffs; n++) {
        double q = 0.;

        for (int m = 0; m <= M; m++)
            q += w[m] * bshat[n][m];

        q *= del;
        coeff[n] += q;
    }
}

/**
   * Computes rho_cen/rem for the coil corresponding to a source point located
   * at z0.  
   */
double KZHCoefficientGenerator<KCoil>::ComputeRho(double z0, bool isCen) const
{
    double r = 0;
    double z = 0;

    if (isCen) {
        r = fCoil->GetR0();

        if (z0 < fCoil->GetZ1() && z0 > fCoil->GetZ0())
            z = 0.;
        else if (fabs(z0 - fCoil->GetZ1()) > fabs(z0 - fCoil->GetZ0()))
            z = fCoil->GetZ0() - z0;
        else
            z = fCoil->GetZ1() - z0;
    }
    else {
        r = fCoil->GetR1();

        if (fabs(z0 - fCoil->GetZ1()) > fabs(z0 - fCoil->GetZ0()))
            z = fCoil->GetZ1() - z0;
        else
            z = fCoil->GetZ0() - z0;
    }

    return sqrt(z * z + r * r);
}

void KZHCoefficientGenerator<KCoil>::GetExtrema(double& zMin, double& zMax) const
{
    zMin = fCoil->GetZ0();
    zMax = fCoil->GetZ1();
}
}  // namespace KEMField

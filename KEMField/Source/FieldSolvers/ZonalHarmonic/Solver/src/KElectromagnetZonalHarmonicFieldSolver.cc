#include "KElectromagnetZonalHarmonicFieldSolver.hh"

#include <numeric>

namespace KEMField
{

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralExpansion(const KPosition& P) const
{
    KFieldVector localP = fContainer.GetCoordinateSystem().ToLocal(P);

    if (UseCentralExpansion(localP))
        return true;

    for (const auto& sub : fSubsetFieldSolvers) {
        if (sub->CentralExpansion(P))
            return true;
    }

    return false;
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteExpansion(const KPosition& P) const
{
    KFieldVector localP = fContainer.GetCoordinateSystem().ToLocal(P);

    if (UseRemoteExpansion(localP))
        return true;

    for (const auto& sub : fSubsetFieldSolvers) {
        if (sub->RemoteExpansion(P))
            return true;
    }

    return false;
}

KFieldVector KZonalHarmonicFieldSolver<KMagnetostaticBasis>::VectorPotential(const KPosition& P) const
{
    KFieldVector localP = fContainer.GetCoordinateSystem().ToLocal(P);

    KFieldVector A;

    if (fCentralFirst) {

        if (UseCentralExpansion(localP)) {
            if (CentralExpansionVectorPotential(localP, A)) {
                fCentralExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(A);
            }
        }

        if (UseRemoteExpansion(localP)) {
            if (RemoteExpansionVectorPotential(localP, A)) {
                fRemoteExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(A);
            }
        }
    }
    else {

        if (UseRemoteExpansion(localP)) {
            if (RemoteExpansionVectorPotential(localP, A)) {
                fRemoteExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(A);
            }
        }

        if (UseCentralExpansion(localP)) {
            if (CentralExpansionVectorPotential(localP, A)) {
                fCentralExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(A);
            }
        }
    }

    if (!fSubsetFieldSolvers.empty()) {
        VectorPotentialAccumulator accumulator(P);
        return std::accumulate(fSubsetFieldSolvers.begin(), fSubsetFieldSolvers.end(), A, accumulator);
    }

    fDirectExecCount++;
    kem_cout(eInfo) << "Magnetic ZH solver falling back to direct integration at point <"
                    << P.Z() << " " << P.Perp() << ">" << eom;
    return fIntegratingFieldSolver.VectorPotential(P);
}

KFieldVector KZonalHarmonicFieldSolver<KMagnetostaticBasis>::MagneticField(const KPosition& P) const
{
    KFieldVector localP = fContainer.GetCoordinateSystem().ToLocal(P);
    KFieldVector B;

    if (fCentralFirst) {

        if (UseCentralExpansion(localP)) {
            if (CentralExpansionMagneticField(localP, B)) {
                fCentralExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(B);
            }
        }

        if (UseRemoteExpansion(localP)) {
            if (RemoteExpansionMagneticField(localP, B)) {
                fRemoteExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(B);
            }
        }
    }
    else {

        if (UseRemoteExpansion(localP)) {
            if (RemoteExpansionMagneticField(localP, B)) {
                fRemoteExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(B);
            }
        }

        if (UseCentralExpansion(localP)) {
            if (CentralExpansionMagneticField(localP, B)) {
                fCentralExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(B);
            }
        }
    }

    if (!fSubsetFieldSolvers.empty()) {
        MagneticFieldAccumulator accumulator(P);
        return std::accumulate(fSubsetFieldSolvers.begin(), fSubsetFieldSolvers.end(), B, accumulator);
    }

    fDirectExecCount++;
    kem_cout(eInfo) << "Magnetic ZH solver falling back to direct integration at point <"
                    << P.Z() << " " << P.Perp() << ">" << eom;
    return fIntegratingFieldSolver.MagneticField(P);
}

KGradient KZonalHarmonicFieldSolver<KMagnetostaticBasis>::MagneticFieldGradient(const KPosition& P) const
{
    KFieldVector localP = fContainer.GetCoordinateSystem().ToLocal(P);
    KGradient g;

    if (fCentralFirst) {

        if (UseCentralExpansion(localP)) {
            if (CentralGradientExpansion(localP, g)) {
                fCentralExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(g);
            }
        }

        if (UseRemoteExpansion(localP)) {
            if (RemoteGradientExpansion(localP, g)) {
                fRemoteExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(g);
            }
        }
    }
    else {

        if (UseRemoteExpansion(localP)) {
            if (RemoteGradientExpansion(localP, g)) {
                fRemoteExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(g);
            }
        }

        if (UseCentralExpansion(localP)) {
            if (CentralGradientExpansion(localP, g)) {
                fCentralExecCount++;
                return fContainer.GetCoordinateSystem().ToGlobal(g);
            }
        }
    }

    if (!fSubsetFieldSolvers.empty()) {
        MagneticFieldGradientAccumulator accumulator(P);
        return std::accumulate(fSubsetFieldSolvers.begin(), fSubsetFieldSolvers.end(), g, accumulator);
    }

    fDirectExecCount++;
    kem_cout(eInfo) << "Magnetic ZH solver falling back to direct integration at point <"
                    << P.Z() << " " << P.Perp() << ">" << eom;
    return fIntegratingFieldSolver.MagneticFieldGradient(P);
}

std::pair<KFieldVector, KGradient>
KZonalHarmonicFieldSolver<KMagnetostaticBasis>::MagneticFieldAndGradient(const KPosition& P) const
{
    KFieldVector localP = fContainer.GetCoordinateSystem().ToLocal(P);

    KFieldVector B;
    KGradient g;


    if (fCentralFirst) {

        if (UseCentralExpansion(localP)) {
            if (CentralMagneticFieldAndGradientExpansion(localP, g, B)) {
                fCentralExecCount++;
                return std::make_pair(fContainer.GetCoordinateSystem().ToGlobal(B),
                                      fContainer.GetCoordinateSystem().ToGlobal(g));
            }
        }

        if (UseRemoteExpansion(localP)) {
            if (RemoteMagneticFieldAndGradientExpansion(localP, g, B)) {
                fRemoteExecCount++;
                return std::make_pair(fContainer.GetCoordinateSystem().ToGlobal(B),
                                      fContainer.GetCoordinateSystem().ToGlobal(g));
            }
        }
    }
    else {

        if (UseRemoteExpansion(localP)) {
            if (RemoteMagneticFieldAndGradientExpansion(localP, g, B)) {
                fRemoteExecCount++;
                return std::make_pair(fContainer.GetCoordinateSystem().ToGlobal(B),
                                      fContainer.GetCoordinateSystem().ToGlobal(g));
            }
        }

        if (UseCentralExpansion(localP)) {
            if (CentralMagneticFieldAndGradientExpansion(localP, g, B)) {
                fCentralExecCount++;
                return std::make_pair(fContainer.GetCoordinateSystem().ToGlobal(B),
                                      fContainer.GetCoordinateSystem().ToGlobal(g));
            }
        }
    }


    if (!fSubsetFieldSolvers.empty()) {
        MagneticFieldAndGradientAccumulator accumulator(P);
        return std::accumulate(fSubsetFieldSolvers.begin(),
                               fSubsetFieldSolvers.end(),
                               std::make_pair(B, g),
                               accumulator);
    }

    fDirectExecCount += 2;
    kem_cout(eInfo) << "Magnetic ZH solver falling back to direct integration at point <"
                    << P.Z() << " " << P.Perp() << ">" << eom;
    B = fIntegratingFieldSolver.MagneticField(P);
    g = fIntegratingFieldSolver.MagneticFieldGradient(P);

    return std::make_pair(B, g);
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralExpansionMagneticField(const KPosition& P,
                                                                                   KFieldVector& B) const
{
    if (fContainer.GetCentralSourcePoints().empty()) {
        B[0] = B[1] = B[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetCentralSourcePoints())[fmCentralSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();

    double proximity_to_sourcepoint = fContainer.GetParameters().GetProximityToSourcePoint();

    // if the field point is very close to the source point
    if (r < proximity_to_sourcepoint && fabs(z - sP.GetZ0()) < proximity_to_sourcepoint) {
        B[0] = B[1] = 0.;
        B[2] = sPcoeff[0];
        return true;
    }

    // rho,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    double u = delz / rho;
    double s = r / rho;

    // Convergence ratio:
    double rc = rho / sP.GetRho();

    // number of precomputed coefficients
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rcn = rc;
    double Bz = sPcoeff[0] + sPcoeff[1] * rc * u;
    double Br = -s * sPcoeff[1] * .5 * rc;

    // flags for series convergence
    bool B_hasConverged = false;

    // n-th Bz, Br terms in the series
    double Bzplus, Brplus;

    // sum of the last 4 B-field terms
    double B_delta[4];
    double B_delta_sum = 0.0;

    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    //Initialize the recursion
    double p1, p1m1, p1m2;
    p1m1 = u;
    p1m2 = 1.;

    double p1p, p1pm1, p1pm2;
    p1pm1 = 1.;
    p1pm2 = 0.;

    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs - 1; n++) {
        p1 = zhc0[n] * u * p1m1 - zhc1[n] * p1m2;
        p1p = zhc2[n] * u * p1pm1 - zhc3[n] * p1pm2;

        rcn *= rc;

        // n-th Bz, Br terms in the series
        Bzplus = sPcoeff[n] * rcn * p1;
        Brplus = -s * sPcoeff[n] * (1.) / (n + 1.) * rcn * p1p;

        Bz += Bzplus;
        Br += Brplus;

        // Conditions for series convergence: these have been changed to match
        // the convergence conditions used by Ferenc's magfield3
        // note that these conditions only enforce convergence on the
        // 'total' magnetic field rather than component wise
        // Also the convergence is only checked after the first 4 terms are computed

        if (n > 5) {
            //  subtract the n-4th term and add the nth one.
            B_delta_sum -= B_delta[n % 4];
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
            if (B_delta_sum < conv_param * (fabs(Bz) + fabs(Br))) {
                B_hasConverged = true;
                break;
            }
        }
        else {
            //add up the B delta sum for the first 4 terms
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
        }

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }

    if (B_hasConverged == false) {
        return false;
    }

    B[2] = Bz;

    if (r < fContainer.GetParameters().GetProximityToSourcePoint())
        B[0] = B[1] = 0.;
    else {
        B[0] = P[0] / r * Br;
        B[1] = P[1] / r * Br;
    }

    return true;
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralExpansionVectorPotential(const KPosition& P,
                                                                                     KFieldVector& A) const
{
    if (fContainer.GetCentralSourcePoints().empty()) {
        A[0] = A[1] = A[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetCentralSourcePoints())[fmCentralSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();
    double sPrho = sP.GetRho();

    double proximity_to_sourcepoint = fContainer.GetParameters().GetProximityToSourcePoint();

    // if the field point is very close to the source point
    if (r < proximity_to_sourcepoint && fabs(z - sP.GetZ0()) < proximity_to_sourcepoint) {
        A[0] = A[1] = A[2] = 0.;
        return true;
    }

    // rho,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    double u = delz / rho;
    double s = r / rho;

    // Convergence ratio:
    double rc = rho / sPrho;

    // number of precomputed coefficients
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rcn = rc;
    double A0 = -s * sPrho * sPcoeff[0] * .5 * rc;

    // flags for series convergence
    bool A_hasConverged = false;

    // n-th A term in the series
    double Aplus;

    // (n-1)-th A term in the series (used for convergence)
    double lastAplus;
    lastAplus = 1.e30;

    // ratio of n-th A term to the series sums
    double A_ratio;

    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    //Initialize the recursion
    double p1p, p1pm1, p1pm2;
    p1pm1 = 1.;
    p1pm2 = 0.;

    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs - 1; n++) {
        p1p = zhc2[n] * u * p1pm1 - zhc3[n] * p1pm2;

        rcn *= rc;

        // n-th A, Bz, Br terms in the series
        Aplus = -sPrho * s * sPcoeff[n - 1] * (1.) / (n * (n + 1.)) * rcn * p1p;

        A0 += Aplus;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        A_ratio = conv_param * fabs(A0);

        if ((fabs(Aplus) < A_ratio && fabs(lastAplus) < A_ratio) || r < proximity_to_sourcepoint) {
            A_hasConverged = true;
            break;
        }

        //update previous terms
        p1pm2 = p1pm1;
        p1pm1 = p1p;

        lastAplus = Aplus;
    }

    if (A_hasConverged == false) {
        return false;
    }

    A[2] = 0;

    if (r < proximity_to_sourcepoint)
        A[0] = A[1] = 0.;
    else {
        A[0] = P[1] / r * A0;
        A[1] = -P[0] / r * A0;
    }

    return true;
}


bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteExpansionMagneticField(const KPosition& P,
                                                                                  KFieldVector& B) const
{
    //      cout <<fmRemoteSPIndex <<endl;
    if (fContainer.GetRemoteSourcePoints().empty()) {
        B[0] = B[1] = B[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetRemoteSourcePoints())[fmRemoteSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();

    // rho,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    if (rho < 1.e-9)
        rho = 1.e-9;
    double u = delz / rho;
    double s = r / rho;

    // Convergence ratio:
    double rr = sP.GetRho() / rho;  // convergence ratio

    // number of precomputed coefficients
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rrn = rr * rr;
    double Bz = 0;
    double Br = 0;

    // flags for series convergence
    bool B_hasConverged = false;

    // n-th A, Bz, Br terms in the series
    double Bzplus, Brplus;

    // sum of the last 4 B-field terms
    double B_delta[4];
    double B_delta_sum = 0.0;

    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    //Initialize the recursion
    double p1, p1m1, p1m2;
    p1m1 = u;
    p1m2 = 1.;

    double p1p, p1pm1, p1pm2;
    p1pm1 = 1.;
    p1pm2 = 0.;

    for (unsigned int n = 2; n < Ncoeffs - 1; n++) {
        //Legendre polynomial recursion relationship
        p1 = zhc0[n] * u * p1m1 - zhc1[n] * p1m2;
        p1p = zhc2[n] * u * p1pm1 - zhc3[n] * p1pm2;

        rrn *= rr;

        // n-th Bz, Br terms in the series
        Bzplus = sPcoeff[n] * rrn * p1;
        Brplus = sPcoeff[n] * s / n * rrn * p1p;

        Bz += Bzplus;
        Br += Brplus;

        // Conditions for series convergence: these have been changed to match
        // the convergence conditions used by Ferenc's magfield3
        // note that these conditions only enforce convergence on the
        // 'total' magnetic field rather than component wise
        // Also the convergence is only checked after the first 4 terms are computed

        if (n > 5) {
            //  subtract the n-4th term and add the nth one.
            B_delta_sum -= B_delta[n % 4];
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
            if (B_delta_sum < conv_param * (fabs(Bz) + fabs(Br))) {
                B_hasConverged = true;
                break;
            }
        }
        else {
            //add up the B delta sum for the first 4 terms
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
        }

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }

    if (!B_hasConverged) {
        return false;
    };

    B[2] = Bz;

    if (r < fContainer.GetParameters().GetProximityToSourcePoint())
        B[0] = B[1] = 0.;
    else {
        B[0] = P[0] / r * Br;
        B[1] = P[1] / r * Br;
    }

    return true;
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteExpansionVectorPotential(const KPosition& P,
                                                                                    KFieldVector& A) const
{
    if (fContainer.GetRemoteSourcePoints().empty()) {
        A[0] = A[1] = A[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetRemoteSourcePoints())[fmRemoteSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();
    double sPrho = sP.GetRho();

    // rho,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    if (rho < 1.e-9)
        rho = 1.e-9;
    double u = delz / rho;
    double s = r / rho;

    // Convergence ratio:
    double rr = sPrho / rho;  // convergence ratio

    // // Create the Legendre polynomial arrays
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rrn = rr * rr;
    double A0 = -sPrho * sPcoeff[2] * s * .5 * rrn;

    // flags for series convergence
    bool A_hasConverged = false;

    // n-th A term in the series
    double Aplus;

    // (n-1)-th A term in the series (used for convergence)
    double lastAplus;
    lastAplus = 1.e30;

    // ratio of n-th A term to the series sums
    double A_ratio;

    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    //Initialize the recursion
    double p1p, p1pm1, p1pm2;
    p1pm1 = 1.;
    p1pm2 = 0.;

    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs - 1; n++) {
        //Legendre polynomial recursion relationship
        p1p = zhc2[n] * u * p1pm1 - zhc3[n] * p1pm2;

        // rrn = (rho_rem/rho)^(n+1)
        rrn *= rr;

        // n-th A, Bz, Br terms in the series
        Aplus = -sPrho * sPcoeff[n + 1] * s * (1.) / (n * (n + 1.)) * rrn * p1p;

        A0 += Aplus;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        A_ratio = conv_param * fabs(A0);

        if ((fabs(Aplus) < A_ratio && fabs(lastAplus) < A_ratio) ||
            r < fContainer.GetParameters().GetProximityToSourcePoint()) {
            A_hasConverged = true;
            break;
        }

        //update previous terms
        p1pm2 = p1pm1;
        p1pm1 = p1p;

        lastAplus = Aplus;
    }

    if (!A_hasConverged)
        return false;

    A[2] = 0;

    if (r < fContainer.GetParameters().GetProximityToSourcePoint())
        A[0] = A[1] = 0.;
    else {
        A[0] = P[1] / r * A0;
        A[1] = -P[0] / r * A0;
    }


    return true;
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralGradientExpansion(const KPosition& P, KGradient& g) const
{
    KFieldVector B(0., 0., 0.);
    return CentralMagneticFieldAndGradientExpansion(P, g, B);
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteGradientExpansion(const KPosition& P, KGradient& g) const
{
    KFieldVector B(0., 0., 0.);
    return RemoteMagneticFieldAndGradientExpansion(P, g, B);
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralMagneticFieldAndGradientExpansion(const KPosition& P,
                                                                                              KGradient& g,
                                                                                              KFieldVector& B) const
{
    if (fContainer.GetCentralSourcePoints().empty()) {
        g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = g[8] = 0.;
        B[0] = B[1] = B[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetCentralSourcePoints())[fmCentralSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();


    // rho,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    double u = delz / rho;
    double u2 = u * u;
    double s = r / rho;
    double s2 = s * s;
    double rho0 = sP.GetRho();

    // Convergence ratio:
    double rc = rho / rho0;

    // number of precomputed coefficients
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // flags for series convergence
    bool B_hasConverged = false;

    // n-th A, Bz, Br terms in the series
    double /*Aplus,*/ Bzplus, Brplus, Bzzplus, Bzrplus, Brrplus;

    // sum of the last 4 B-field terms
    double B_delta[4];
    double B_delta_sum = 0.0;

    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    // first polynoms
    double p1p1, p1, p1m1;
    p1m1 = u;
    p1 = zhc0[2] * u * p1m1 - zhc1[2];

    double p1pp1, p1p, p1pm1;
    p1pm1 = 1.;
    p1p = zhc2[2] * u * p1pm1;

    // First 2 terms of the series:
    double rcn = rc;
    //    double A = -s*rho0*sPcoeff[0]*.5*rc;
    double Bz = sPcoeff[0] + sPcoeff[1] * rc * u;
    double Br = -s * sPcoeff[1] * .5 * rc;
    double Bzz = sPcoeff[1] / rho0;
    double Bzr = 0.;
    double Brr = -sPcoeff[1] * .5 / (rho0 * s2) * ((s2 - 2. * u2) + (s2 + 4. * u2) * u2 - 2. * u2 * p1);


    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs - 2; n++) {
        p1p1 = zhc0[n + 1] * u * p1 - zhc1[n + 1] * p1m1;
        p1pp1 = zhc2[n + 1] * u * p1p - zhc3[n + 1] * p1pm1;

        rcn *= rc;

        // n-th A, Bz, Br terms in the series
        //      Aplus=-rho0*s*sPcoeff[n-1]*(1.)/(n*(n+1.))*rcn*p1p;
        Bzplus = sPcoeff[n] * rcn * p1;
        Brplus = -s * sPcoeff[n] * (1.) / (n + 1.) * rcn * p1p;
        Bzzplus = sPcoeff[n] * rcn * n / rho * p1m1;
        Bzrplus = -s * sPcoeff[n] * rcn / rho * p1pm1;
        Brrplus = -sPcoeff[n] * (n / (n + 1.)) * rcn / (rho * s2) *
                  ((n * s2 - (n + 1.) * u2) * p1m1 + u * (s2 + 2. * (n + 1.) * u2) * p1 - (n + 1.) * u2 * p1p1);

        //      A+=Aplus;
        Bz += Bzplus;
        Br += Brplus;
        Bzz += Bzzplus;
        Bzr += Bzrplus;
        Brr += Brrplus;

        // Conditions for series convergence: these have been changed to match
        // the convergence conditions used by Ferenc's magfield3
        // note that these conditions only enforce convergence on the
        // 'total' magnetic field rather than component wise
        // Also the convergence is only checked after the first 4 terms are computed

        if (n > 5) {
            //  subtract the n-4th term and add the nth one.
            B_delta_sum -= B_delta[n % 4];
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
            if (B_delta_sum < conv_param * (fabs(Bz) + fabs(Br))) {
                B_hasConverged = true;
                break;
            }
        }
        else {
            //add up the B delta sum for the first 4 terms
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
        }

        //update previous terms
        p1m1 = p1;
        p1 = p1p1;
        p1pm1 = p1p;
        p1p = p1pp1;
    }

    if (B_hasConverged == false) {
        return false;
    }

    if (r < fContainer.GetParameters().GetProximityToSourcePoint()) {
        //when particle is on or near axis of symmetry, to avoid numerical
        //instability in calculating the azimuthal angle we use a numerical derivative
        //to calculate the gradient
        B[0] = B[1] = 0.;
        g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = g[8] = 0.;
        CentralGradientExpansionNumerical(P, g);
    }
    else {
        double cosine = P[0] / r;
        double sine = P[1] / r;
        g[0] = Brr * cosine * cosine + 1. / r * Br * sine * sine;
        g[1] = g[3] = Brr * cosine * sine - 1. / r * Br * cosine * sine;
        g[2] = Bzr * cosine;
        g[4] = Brr * sine * sine + 1. / r * Br * cosine * cosine;
        g[5] = Bzr * sine;
        g[6] = Bzr * cosine;
        g[7] = Bzr * sine;
        g[8] = Bzz;

        B[0] = cosine * Br;
        B[1] = sine * Br;
    }
    B[2] = Bz;

    return true;
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteMagneticFieldAndGradientExpansion(const KPosition& P,
                                                                                             KGradient& g,
                                                                                             KFieldVector& B) const
{
    if (fContainer.GetRemoteSourcePoints().empty()) {
        g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = g[8] = 0.;
        B[0] = B[1] = B[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetRemoteSourcePoints())[fmRemoteSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();

    // rho,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    if (rho < 1.e-9)
        rho = 1.e-9;
    double u = delz / rho;
    double s = r / rho;
    double s2 = s * s;

    // Convergence ratio:
    double rr = sP.GetRho() / rho;  // convergence ratio

    // number of precomputed coefficients
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rrn = rr * rr;
    //    double A = -sP.GetRho()*sPcoeff[2]*s*.5*rrn;
    double Bz = 0;
    double Br = 0;
    double Bzz = 0;
    double Bzr = 0;
    double Brr = 0;

    // flags for series convergence
    bool B_hasConverged = false;

    // n-th A, Bz, Br terms in the series
    double /*Aplus,*/ Bzplus, Brplus, Bzzplus, Bzrplus, Brrplus;

    // sum of the last 4 B-field terms
    double B_delta[4];
    double B_delta_sum = 0.0;

    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    // first polynoms
    double p1p1, p1, p1m1;
    p1m1 = u;
    p1 = zhc0[2] * u * p1m1 - zhc1[2];

    double p1pp1, p1p, p1pm1;
    p1pm1 = 1.;
    p1p = zhc2[2] * u * p1pm1;

    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs - 2; n++) {
        p1p1 = zhc0[n + 1] * u * p1 - zhc1[n + 1] * p1m1;
        p1pp1 = zhc2[n + 1] * u * p1p - zhc3[n + 1] * p1pm1;

        rrn *= rr;

        // n-th A, Bz, Br terms in the series
        //      Aplus=-sP.GetRho()*sPcoeff[n+1]*s*(1.)/(n*(n+1.))*rrn*p1p;
        Bzplus = sPcoeff[n] * rrn * p1;
        Brplus = sPcoeff[n] * s / n * rrn * p1p;
        Bzzplus = -sPcoeff[n] * (n + 1.) * rrn / rho * p1p1;
        Bzrplus = -sPcoeff[n] * rrn * s / rho * p1pp1;
        Brrplus = sPcoeff[n] * rrn / rho * ((n + 1) * u * p1 - (s2 + 1. / n) * p1p);

        //      A+=Aplus;
        Bz += Bzplus;
        Br += Brplus;
        Bzz += Bzzplus;
        Bzr += Bzrplus;
        Brr += Brrplus;

        // Conditions for series convergence: these have been changed to match
        // the convergence conditions used by Ferenc's magfield3
        // note that these conditions only enforce convergence on the
        // 'total' magnetic field rather than component wise
        // Also the convergence is only checked after the first 4 terms are computed

        if (n > 5) {
            //  subtract the n-4th term and add the nth one.
            B_delta_sum -= B_delta[n % 4];
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
            if (B_delta_sum < conv_param * (fabs(Bz) + fabs(Br))) {
                B_hasConverged = true;
                break;
            }
        }
        else {
            //add up the B delta sum for the first 4 terms
            B_delta[n % 4] = fabs(Bzplus) + fabs(Brplus);
            B_delta_sum += B_delta[n % 4];
        }

        //update previous terms
        p1m1 = p1;
        p1 = p1p1;
        p1pm1 = p1p;
        p1p = p1pp1;
    }

    if (B_hasConverged == false) {
        return false;
    }


    B[2] = Bz;

    if (r < fContainer.GetParameters().GetProximityToSourcePoint()) {
        //when particle is on or near axis of symmetry, to avoid numerical
        //instability in calculating the azimuthal angle we use a numerical derivative
        //to calculate the gradient
        g[0] = g[1] = g[2] = g[3] = g[4] = g[5] = g[6] = g[7] = g[8] = 0.;
        B[0] = B[1] = 0.;
        RemoteGradientExpansionNumerical(P, g);
    }
    else {
        double cosine = P[0] / r;
        double sine = P[1] / r;
        g[0] = Brr * cosine * cosine + 1. / r * Br * sine * sine;
        g[1] = g[3] = Brr * cosine * sine - 1. / r * Br * cosine * sine;
        g[2] = Bzr * cosine;
        g[4] = Brr * sine * sine + 1. / r * Br * cosine * cosine;
        g[5] = Bzr * sine;
        g[6] = Bzr * cosine;
        g[7] = Bzr * sine;
        g[8] = Bzz;

        B[0] = cosine * Br;
        B[1] = sine * Br;
    }

    return true;
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::CentralGradientExpansionNumerical(const KPosition& P,
                                                                                       KGradient& g) const
{
    //slow, brute force, three-point numerical differentiation
    double num_eps = 1e-6;
    bool retval = true;
    KFieldVector coord_ax[3];
    KFieldVector result[3];
    coord_ax[0] = KFieldVector(1, 0, 0);
    coord_ax[1] = KFieldVector(0, 1, 0);
    coord_ax[2] = KFieldVector(0, 0, 1);

    for (int i = 0; i < 3; i++) {
        KFieldVector forward;
        KFieldVector backward;
        retval = retval && CentralExpansionMagneticField(P + num_eps * coord_ax[i], forward);
        retval = retval && CentralExpansionMagneticField(P - num_eps * coord_ax[i], backward);
        result[i] = (forward - backward) / (2.0 * num_eps);
        g(0, i) = result[i][0];
        g(1, i) = result[i][1];
        g(2, i) = result[i][2];
    }

    return retval;
}

bool KZonalHarmonicFieldSolver<KMagnetostaticBasis>::RemoteGradientExpansionNumerical(const KPosition& P,
                                                                                      KGradient& g) const
{
    //slow, brute force, three-point numerical differentiation
    double num_eps = 1e-6;
    bool retval = true;
    KFieldVector coord_ax[3];
    KFieldVector result[3];
    coord_ax[0] = KFieldVector(1, 0, 0);
    coord_ax[1] = KFieldVector(0, 1, 0);
    coord_ax[2] = KFieldVector(0, 0, 1);

    for (int i = 0; i < 3; i++) {
        KFieldVector forward;
        KFieldVector backward;
        retval = retval && RemoteExpansionMagneticField(P + num_eps * coord_ax[i], forward);
        retval = retval && RemoteExpansionMagneticField(P - num_eps * coord_ax[i], backward);
        result[i] = (forward - backward) / (2.0 * num_eps);
        g(0, i) = result[i][0];
        g(1, i) = result[i][1];
        g(2, i) = result[i][2];
    }

    return retval;
}


}  // namespace KEMField

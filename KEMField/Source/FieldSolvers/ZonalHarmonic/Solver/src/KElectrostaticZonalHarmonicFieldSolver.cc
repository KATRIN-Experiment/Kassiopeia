#include "KElectrostaticZonalHarmonicFieldSolver.hh"

#include "KEMCoreMessage.hh"

#include <numeric>

// #include "KShanksTransformation.hh"

namespace KEMField
{
bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::CentralExpansion(const KPosition& P) const
{
    if (UseCentralExpansion(P))
        return true;

    for (const auto& sub : fSubsetFieldSolvers) {
        if (sub->CentralExpansion(P))
            return true;
    }

    return false;
}

bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::RemoteExpansion(const KPosition& P) const
{
    if (UseRemoteExpansion(P))
        return true;

    for (const auto& sub : fSubsetFieldSolvers) {
        if (sub->UseRemoteExpansion(P))
            return true;
    }

    return false;
}

double KZonalHarmonicFieldSolver<KElectrostaticBasis>::Potential(const KPosition& P) const
{
    double phi = 0;

    if (UseCentralExpansion(P))
        if (CentralExpansionPotential(P, phi))
            return phi;

    if (UseRemoteExpansion(P))
        if (RemoteExpansionPotential(P, phi))
            return phi;

    if (!fSubsetFieldSolvers.empty()) {
        PotentialAccumulator accumulator(P);

        return std::accumulate(fSubsetFieldSolvers.begin(), fSubsetFieldSolvers.end(), phi, accumulator);
    }

    kem_cout_debug("ZH solver falling back to direct integration at point <" << P.Z() << " " << P.Perp() << ">"   << eom);
    return fIntegratingFieldSolver.Potential(P);
}

KFieldVector KZonalHarmonicFieldSolver<KElectrostaticBasis>::ElectricField(const KPosition& P) const
{
    KFieldVector E;

    if (UseCentralExpansion(P))
        if (CentralExpansionField(P, E))
            return E;

    if (UseRemoteExpansion(P))
        if (RemoteExpansionField(P, E))
            return E;

    if (!fSubsetFieldSolvers.empty()) {
        ElectricFieldAccumulator accumulator(P);

        return std::accumulate(fSubsetFieldSolvers.begin(), fSubsetFieldSolvers.end(), E, accumulator);
    }

    kem_cout_debug("ZH solver falling back to direct integration at point <" << P.Z() << " " << P.Perp() << ">"   << eom);
    return fIntegratingFieldSolver.ElectricField(P);
}

std::pair<KFieldVector, double>
KZonalHarmonicFieldSolver<KElectrostaticBasis>::ElectricFieldAndPotential(const KPosition& P) const
{
    KFieldVector E;
    double phi = 0;

    if (UseCentralExpansion(P))
        if (CentralExpansionFieldAndPotential(P, E, phi))
            return std::make_pair(E, phi);

    if (UseRemoteExpansion(P))
        if (RemoteExpansionFieldAndPotential(P, E, phi))
            return std::make_pair(E, phi);

    if (!fSubsetFieldSolvers.empty()) {
        ElectricFieldAndPotentialAccumulator accumulator(P);

        return std::accumulate(fSubsetFieldSolvers.begin(),
                               fSubsetFieldSolvers.end(),
                               std::make_pair(E, phi),
                               accumulator);
    }

    kem_cout_debug("ZH solver falling back to direct integration at point <" << P.Z() << " " << P.Perp() << ">"   << eom);
    return std::make_pair(fIntegratingFieldSolver.ElectricField(P), fIntegratingFieldSolver.Potential(P));
}

bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::CentralExpansionPotential(const KPosition& P,
                                                                               double& potential) const
{
    if (fContainer.GetCentralSourcePoints().empty()) {
        potential = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetCentralSourcePoints())[fmCentralSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();
    double sPrho = sP.GetRho();

    // if the field point is very close to the source point
    if (r < fContainer.GetParameters().GetProximityToSourcePoint() &&
        fabs(z - sP.GetZ0()) < fContainer.GetParameters().GetProximityToSourcePoint()) {
        potential = sP.GetCoeff(0);
        return true;
    }

    //retrieve the legendre polynomial coefficients
    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    // ro,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    double u = delz / rho;

    // Convergence ratio:
    double rc = rho / sPrho;

    //number of Legendre polynomials
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rcn = rc;
    double Phi = sPcoeff[0] + sPcoeff[1] * rc * u;

    // flags for series convergence
    bool Phi_hasConverged = false;

    // n-th Phi, Ez, Er terms in the series
    double Phiplus;

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double lastPhiplus;
    lastPhiplus = 1.e30;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Phi_ratio;

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

        // rcn = (rho/rho_cen)^n
        rcn *= rc;

        // n-th Phi, Ez, Er terms in the series
        Phiplus = sPcoeff[n] * rcn * p1;

        Phi += Phiplus;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        Phi_ratio = conv_param * fabs(Phi);

        if (fabs(Phiplus) < Phi_ratio && fabs(lastPhiplus) < Phi_ratio)
            Phi_hasConverged = true;

        if (Phi_hasConverged == true)
            break;

        lastPhiplus = Phiplus;

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }

    if (Phi_hasConverged == false)
        return false;

    potential = Phi;

    return true;
}

bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::CentralExpansionField(const KPosition& P,
                                                                           KFieldVector& electricField) const
{
    if (fContainer.GetCentralSourcePoints().empty()) {
        electricField[0] = electricField[1] = electricField[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetCentralSourcePoints())[fmCentralSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();
    double sPrho = sP.GetRho();

    // if the field point is very close to the source point
    if (r < fContainer.GetParameters().GetProximityToSourcePoint() &&
        fabs(z - sP.GetZ0()) < fContainer.GetParameters().GetProximityToSourcePoint()) {
        electricField[2] = -sP.GetCoeff(1) / sP.GetRho();
        electricField[0] = electricField[1] = 0;
        return true;
    }

    //retrieve the legendre polynomial coefficients
    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();
    double prox_param = fContainer.GetParameters().GetProximityToSourcePoint();

    // ro,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    double u = delz / rho;
    double s = r / rho;

    // Convergence ratio:
    double rc = rho / sPrho;

    //number of Legendre polynomials
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rcn = rc;
    double Ez = (sPcoeff[1] + sPcoeff[2] * 2. * rc * u);
    double Er = sPcoeff[2] * rc;

    // flags for series convergence
    bool Ez_hasConverged = false;
    bool Er_hasConverged = false;

    // n-th Phi, Ez, Er terms in the series
    double Ezplus, Erplus;

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double lastEzplus, lastErplus;
    lastEzplus = lastErplus = 1.e30;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Ez_ratio, Er_ratio;

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

        // rcn = (rho/rho_cen)^n
        rcn *= rc;

        // n-th Phi, Ez, Er terms in the series
        Ezplus = sPcoeff[n + 1] * (n + 1.) * rcn * p1;
        Erplus = sPcoeff[n + 1] * rcn * p1p;

        Ez += Ezplus;
        Er += Erplus;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        Ez_ratio = conv_param * fabs(Ez);
        Er_ratio = conv_param * fabs(Er);

        if (fabs(Ezplus) < Ez_ratio && fabs(lastEzplus) < Ez_ratio)
            Ez_hasConverged = true;
        if ((fabs(Erplus) < Er_ratio && fabs(lastErplus) < Er_ratio) || r < prox_param)
            Er_hasConverged = true;

        if (Ez_hasConverged * Er_hasConverged == true)
            break;

        lastEzplus = Ezplus;
        lastErplus = Erplus;

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }

    if (Ez_hasConverged * Er_hasConverged == false)
        return false;

    Ez *= -1. / sPrho;
    Er *= s / sPrho;

    electricField[2] = Ez;

    if (r < prox_param)
        electricField[0] = electricField[1] = 0.;
    else {
        double cosine = P[0] / r;
        double sine = P[1] / r;
        electricField[0] = cosine * Er;
        electricField[1] = sine * Er;
    }

    return true;
}

bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::CentralExpansionFieldAndPotential(const KPosition& P,
                                                                                       KFieldVector& electricField,
                                                                                       double& potential) const
{
    if (fContainer.GetCentralSourcePoints().empty()) {
        potential = 0.;
        electricField[0] = electricField[1] = electricField[2] = 0.;
        return true;
    }

    double r = sqrt(P[0] * P[0] + P[1] * P[1]);
    double z = P[2];

    const KZonalHarmonicSourcePoint& sP = *((fContainer.GetCentralSourcePoints())[fmCentralSPIndex]);
    const double* sPcoeff = sP.GetRawPointerToCoeff();
    double sPrho = sP.GetRho();

    // if the field point is very close to the source point
    if (r < fContainer.GetParameters().GetProximityToSourcePoint() &&
        fabs(z - sP.GetZ0()) < fContainer.GetParameters().GetProximityToSourcePoint()) {
        electricField[2] = -sP.GetCoeff(1) / sP.GetRho();
        electricField[0] = electricField[1] = 0;
        potential = sP.GetCoeff(0);
        return true;
    }

    //retrieve the legendre polynomial coefficients
    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();
    double prox_param = fContainer.GetParameters().GetProximityToSourcePoint();

    // ro,u,s:
    double delz = z - sP.GetZ0();
    double rho = sqrt(r * r + delz * delz);
    double u = delz / rho;
    double s = r / rho;

    // Convergence ratio:
    double rc = rho / sPrho;

    //number of Legendre polynomials
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // First 2 terms of the series:
    double rcn = rc;
    double Ez = (sPcoeff[1] + sPcoeff[2] * 2. * rc * u);
    double Er = sPcoeff[2] * rc;
    double Phi = sPcoeff[0] + sPcoeff[1] * rc * u;

    // flags for series convergence
    bool Phi_hasConverged = false;
    bool Ez_hasConverged = false;
    bool Er_hasConverged = false;

    // n-th Phi, Ez, Er terms in the series
    double Phiplus, Ezplus, Erplus;

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double lastPhiplus, lastEzplus, lastErplus;
    lastPhiplus = lastEzplus = lastErplus = 1.e30;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Phi_ratio, Ez_ratio, Er_ratio;

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

        // rcn = (rho/rho_cen)^n
        rcn *= rc;

        // n-th Phi, Ez, Er terms in the series
        Ezplus = sPcoeff[n + 1] * (n + 1.) * rcn * p1;
        Erplus = sPcoeff[n + 1] * rcn * p1p;
        Phiplus = sPcoeff[n] * rcn * p1;

        Phi += Phiplus;
        Ez += Ezplus;
        Er += Erplus;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        Phi_ratio = conv_param * fabs(Phi);
        Ez_ratio = conv_param * fabs(Ez);
        Er_ratio = conv_param * fabs(Er);

        if (fabs(Phiplus) < Phi_ratio && fabs(lastPhiplus) < Phi_ratio)
            Phi_hasConverged = true;
        if (fabs(Ezplus) < Ez_ratio && fabs(lastEzplus) < Ez_ratio)
            Ez_hasConverged = true;
        if ((fabs(Erplus) < Er_ratio && fabs(lastErplus) < Er_ratio) || r < prox_param)
            Er_hasConverged = true;

        if (Phi_hasConverged * Ez_hasConverged * Er_hasConverged == true)
            break;

        lastPhiplus = Phiplus;
        lastEzplus = Ezplus;
        lastErplus = Erplus;

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }

    if (Phi_hasConverged * Ez_hasConverged * Er_hasConverged == false)
        return false;

    Ez *= -1. / sPrho;
    Er *= s / sPrho;

    potential = Phi;
    electricField[2] = Ez;

    if (r < prox_param)
        electricField[0] = electricField[1] = 0.;
    else {
        electricField[0] = P[0] / r * Er;
        electricField[1] = P[1] / r * Er;
    }

    return true;
}

bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::RemoteExpansionPotential(const KPosition& P,
                                                                              double& potential) const
{
    if (fContainer.GetRemoteSourcePoints().empty()) {
        potential = 0.;
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

    // Convergence ratio:
    double rr = sPrho / rho;  // convergence ratio

    //retrieve the legendre polynomial coefficients
    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();

    //number of legendre polynomials
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // series loop starts at n = 2, so we manually compute the first three terms

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double Phi_n_1 = sPcoeff[0] * rr;

    // n-th Phi, Ez, Er terms in the series
    double rrn = rr * rr;
    double Phi_n = sPcoeff[1] * rrn * u;

    // First 3 terms of the series:
    double Phi = Phi_n_1 + Phi_n;

    // flags for series convergence
    bool Phi_hasConverged = false;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Phi_ratio;

    //Initialize the recursion
    double p1, p1m1, p1m2;
    p1m1 = u;
    p1m2 = 1.;
    double p1p, p1pm1, p1pm2;
    p1pm1 = 1.;
    p1pm2 = 0.;

    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs; n++) {
        p1 = zhc0[n] * u * p1m1 - zhc1[n] * p1m2;
        p1p = zhc2[n] * u * p1pm1 - zhc3[n] * p1pm2;

        // rrn = (rho_rem/rho)^(n+1)
        rrn *= rr;

        // n-th Phi, Ez, Er terms in the series
        Phi_n = sPcoeff[n] * rrn * p1;

        Phi += Phi_n;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        Phi_ratio = conv_param * fabs(Phi);

        if (fabs(Phi_n) < Phi_ratio && fabs(Phi_n_1) < Phi_ratio)
            Phi_hasConverged = true;


        if (Phi_hasConverged == true)
            break;

        Phi_n_1 = Phi_n;

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }

    if (Phi_hasConverged == false)
        return false;

    potential = Phi;

    return true;
}


bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::RemoteExpansionField(const KPosition& P,
                                                                          KFieldVector& electricField) const
{
    if (fContainer.GetRemoteSourcePoints().empty()) {
        electricField[0] = electricField[1] = electricField[2] = 0.;
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

    //retrieve the legendre polynomial coefficients
    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();
    double prox_param = fContainer.GetParameters().GetProximityToSourcePoint();

    //number of legendre polynomials
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // series loop starts at n = 2, so we manually compute the first three terms

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double Ez_n_1 = 0.;
    double Er_n_1 = 0.;

    // n-th Phi, Ez, Er terms in the series
    double rrn = rr * rr;
    double Ez_n = sPcoeff[0] * rrn * u;
    double Er_n = sPcoeff[0] * rrn;

    // First 3 terms of the series:
    double Ez = Ez_n_1 + Ez_n;
    double Er = Er_n_1 + Er_n;

    // flags for series convergence
    bool Ez_hasConverged = false;
    bool Er_hasConverged = false;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Ez_ratio, Er_ratio;

    //Initialize the recursion
    double p1, p1m1, p1m2;
    p1m1 = u;
    p1m2 = 1.;
    double p1p, p1pm1, p1pm2;
    p1pm1 = 1.;
    p1pm2 = 0.;

    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs; n++) {
        p1 = zhc0[n] * u * p1m1 - zhc1[n] * p1m2;
        p1p = zhc2[n] * u * p1pm1 - zhc3[n] * p1pm2;

        // rrn = (rho_rem/rho)^(n+1)
        rrn *= rr;

        // n-th Phi, Ez, Er terms in the series
        Ez_n = sPcoeff[n - 1] * n * rrn * p1;
        Er_n = sPcoeff[n - 1] * rrn * p1p;

        Ez += Ez_n;
        Er += Er_n;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        Ez_ratio = conv_param * fabs(Ez);
        Er_ratio = conv_param * fabs(Er);

        if (fabs(Ez_n) < Ez_ratio && fabs(Ez_n_1) < Ez_ratio)
            Ez_hasConverged = true;
        if ((fabs(Er_n) < Er_ratio && fabs(Er_n_1) < Er_ratio) || r < prox_param)
            Er_hasConverged = true;

        if (Ez_hasConverged * Er_hasConverged == true)
            break;

        Ez_n_1 = Ez_n;
        Er_n_1 = Er_n;

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }

    if (Ez_hasConverged * Er_hasConverged == false)
        return false;

    Ez *= 1. / sPrho;
    Er *= s / sPrho;

    electricField[2] = Ez;

    if (r < prox_param)
        electricField[0] = electricField[1] = 0;
    else {
        double cosine = P[0] / r;
        double sine = P[1] / r;
        electricField[0] = cosine * Er;
        electricField[1] = sine * Er;
    }

    return true;
}

bool KZonalHarmonicFieldSolver<KElectrostaticBasis>::RemoteExpansionFieldAndPotential(const KPosition& P,
                                                                                      KFieldVector& electricField,
                                                                                      double& potential) const
{
    if (fContainer.GetRemoteSourcePoints().empty()) {
        electricField[0] = electricField[1] = electricField[2] = 0.;
        potential = 0.;
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

    //retrieve the legendre polynomial coefficients
    const double* zhc0 = fZHCoeffSingleton->GetRawPointerToRow(0);
    const double* zhc1 = fZHCoeffSingleton->GetRawPointerToRow(1);
    const double* zhc2 = fZHCoeffSingleton->GetRawPointerToRow(2);
    const double* zhc3 = fZHCoeffSingleton->GetRawPointerToRow(3);
    double conv_param = fContainer.GetParameters().GetConvergenceParameter();
    double prox_param = fContainer.GetParameters().GetProximityToSourcePoint();

    //number of legendre polynomials
    unsigned int Ncoeffs = sP.GetNCoeffs();

    // series loop starts at n = 2, so we manually compute the first three terms

    // (n-1)-th Phi, Ez, Er terms in the series (used for convergence)
    double Phi_n_1 = sPcoeff[0] * rr;
    double Ez_n_1 = 0.;
    double Er_n_1 = 0.;

    // n-th Phi, Ez, Er terms in the series
    double rrn = rr * rr;
    double Phi_n = sPcoeff[1] * rrn * u;
    double Ez_n = sPcoeff[0] * rrn * u;
    double Er_n = sPcoeff[0] * rrn;

    // First 3 terms of the series:
    double Phi = Phi_n_1 + Phi_n;
    double Ez = Ez_n_1 + Ez_n;
    double Er = Er_n_1 + Er_n;

    // flags for series convergence
    bool Phi_hasConverged = false;
    bool Ez_hasConverged = false;
    bool Er_hasConverged = false;

    // ratio of n-th Phi, Ez, Er terms to the series sums
    double Phi_ratio, Ez_ratio, Er_ratio;

    //Initialize the recursion
    double p1, p1m1, p1m2;
    p1m1 = u;
    p1m2 = 1.;
    double p1p, p1pm1, p1pm2;
    p1pm1 = 1.;
    p1pm2 = 0.;

    // Compute the series expansion
    for (unsigned int n = 2; n < Ncoeffs; n++) {
        p1 = zhc0[n] * u * p1m1 - zhc1[n] * p1m2;
        p1p = zhc2[n] * u * p1pm1 - zhc3[n] * p1pm2;

        // rrn = (rho_rem/rho)^(n+1)
        rrn *= rr;

        // n-th Phi, Ez, Er terms in the series
        Phi_n = sPcoeff[n] * rrn * p1;
        Ez_n = sPcoeff[n - 1] * n * rrn * p1;
        Er_n = sPcoeff[n - 1] * rrn * p1p;

        Phi += Phi_n;
        Ez += Ez_n;
        Er += Er_n;

        // Conditions for series convergence:
        //   the last term in the series must be smaller than the current series
        //   sum by the given parameter, and smaller than the previous term
        Phi_ratio = conv_param * fabs(Phi);
        Ez_ratio = conv_param * fabs(Ez);
        Er_ratio = conv_param * fabs(Er);

        if (fabs(Phi_n) < Phi_ratio && fabs(Phi_n_1) < Phi_ratio)
            Phi_hasConverged = true;
        if (fabs(Ez_n) < Ez_ratio && fabs(Ez_n_1) < Ez_ratio)
            Ez_hasConverged = true;
        if ((fabs(Er_n) < Er_ratio && fabs(Er_n_1) < Er_ratio) || r < prox_param)
            Er_hasConverged = true;

        if (Phi_hasConverged * Ez_hasConverged * Er_hasConverged == true)
            break;

        Phi_n_1 = Phi_n;
        Ez_n_1 = Ez_n;
        Er_n_1 = Er_n;

        //update previous terms
        p1m2 = p1m1;
        p1m1 = p1;
        p1pm2 = p1pm1;
        p1pm1 = p1p;
    }


    if (Phi_hasConverged * Ez_hasConverged * Er_hasConverged == false)
        return false;

    Ez *= 1. / sPrho;
    Er *= s / sPrho;

    potential = Phi;
    electricField[2] = Ez;

    if (r < prox_param)
        electricField[0] = electricField[1] = 0;
    else {
        double cosine = P[0] / r;
        double sine = P[1] / r;
        electricField[0] = cosine * Er;
        electricField[1] = sine * Er;
    }

    return true;
}
}  // namespace KEMField

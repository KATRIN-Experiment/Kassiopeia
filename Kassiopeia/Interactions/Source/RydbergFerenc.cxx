//
// Created by trost on 26.05.15.
//

#include "RydbergFerenc.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSInteractionsMessage.h"
#include "QuadGaussLegendre.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
using katrin::KRandom;

#include <numeric>


namespace Kassiopeia
{

FBBRionization::FBBRionization(double aT, int an, int al) : fT(aT), fn(an), fl(al)
{
    for (int k = 1; k <= 2 * nMAX + 9; k++)
        LogN[k] = log((double) k);
    LogN[0] = 0.;
    LogNFactor[0] = 0;
    LogNFactor[1] = 0;
    for (int k = 2; k <= 2 * nMAX + 9; k++)
        LogNFactor[k] = LogNFactor[k - 1] + LogN[k];
    Ehigh = log(std::numeric_limits<double>::max()) * 0.25;
    Chigh = exp(Ehigh);
    Clow = exp(-Ehigh);
    logpi = log(M_PI);
    fC = 4. * M_PI * M_PI * katrin::KConst::Alpha() / 3.;
    fAtomic_C = 1. / katrin::KConst::Alpha();
    fAtomic_kB = 3.1668e-6;
}

/////////////////////////////////////////////////////////////////////

FBBRionization::~FBBRionization() {}

/////////////////////////////////////////////////////////////////////

double FBBRionization::RadInt2BoundFreeBurgess(int n, int l, double E, int sign)
{

    int lp = l + sign;
    if (lp < 0)
        return 0.;

    if (n < 1 || l < 0 || l > n - 1) {
        puts("Message from function RadInt2BoundFreeBurgess:");
        puts("n has to be larger than 0!");
        puts("l has to be larger than -1 and smaller than n !");
        puts("Program running is stopped !!! ");
        exit(1);
    }

    if (n > nMAX) {
        puts("Message from function RadInt2BoundFreeBurgess");
        puts("n cannot be larger than nMAX!");
        puts("Program running is stopped !!! ");
        exit(1);
    }

    if (E < 0.) {
        puts("Message from function RadInt2BoundFreeComplex");
        puts("The outgoing electron energy E has to be positive!");
        puts("Program running is stopped !!! ");
        exit(1);
    }

    if (sign != -1 && sign != +1) {
        puts("Message from function RadInt2BoundFreeBurgess:");
        puts("sign has to be either -1 or +1!");
        puts("Program running is stopped !!! ");
        exit(1);
    }

    double k = sqrt(2. * fabs(E));  // free electron momentum (velocity) in atomic units

    if (k < 1.e-10)  // k=0 would cause division by zero error
    {
        k = 1.e-10;
    }
    double k2 = k * k;
    double D = 1. + n * n * k2;

    // Eq. 30 of Burgess 1965:
    double logg0 = 0.5 * (logpi - LogN[2] - LogNFactor[2 * n - 1]) + LogN[4] + n * (LogN[4] + LogN[n]) - 2 * n;
    //  Eq. 31 of Burgess 1965:
    double logPs = 0.;
    for (int s = 1; s <= n; s++) {
        logPs += log(1. + s * s * k2);
    }
    double logg1 =
        0.5 * (logPs - log(1. - exp(-2. * M_PI / k))) + 2 * n - 2. / k * atan(n * k) - (n + 2) * log(D) + logg0;

    double g1 = 1.;  // we use here g1=1 instead of exp(logg1), to avoid overflow or underflow
    //  Eq. 32 of Burgess 1965:
    double g2 = 0.5 * sqrt((2. * n - 1.) * D) * g1;
    //  Eq. 33 of Burgess 1965:
    double g3 = 1. / (2 * n) * sqrt(D / (1. + (n - 1.) * (n - 1.) * k2)) * g1;
    //  Eq. 34 of Burgess 1965:
    double g4 = (4. + (n - 1.) * D) / (2. * n) * sqrt((2. * n - 1.) / (1. + (n - 2.) * (n - 2.) * k2)) * g3;

    //    printf("logg0,logg1= %12.3f %12.3f  \t\n",logg0,logg1);

    double Exponent = logg1;

    double g = 0., RadInt;

    // We have to multiply the output by 2/M_PI, due to the different wave func. normalizations;
    // Burgess uses a normalization of pi*delta(k^2-k'^2), and I use delta(E-E');
    // therefore my free wave function is sqrt(2./M_PI) times larger than the free wave func. of Burgess.

    double Cfac = 2. / M_PI;

    if (l == n - 1 && sign == 1) {
        g = g1 * exp(Exponent);
        RadInt = n * n * g;
        return RadInt * RadInt * Cfac;
    }
    else if (l == n - 2 && sign == 1) {
        g = g2 * exp(Exponent);
        RadInt = n * n * g;
        return RadInt * RadInt * Cfac;
    }
    else if (l == n - 1 && sign == -1) {
        g = g3 * exp(Exponent);
        RadInt = n * n * g;
        return RadInt * RadInt * Cfac;
    }
    else if (l == n - 2 && sign == -1) {
        g = g4 * exp(Exponent);
        RadInt = n * n * g;
        return RadInt * RadInt * Cfac;
    }

    double A, B, C;
    int j;


    if (sign == 1) {
        for (int L = n - 3; L >= l; L--) {
            j = L + 2;
            A = 2. * n * sqrt((n * n - (j - 1) * (j - 1)) * (1. + j * j * k2));
            B = 4. * n * n - 4. * j * j + j * (2 * j - 1) * D;
            C = -2. * n * sqrt((n * n - j * j) * (1. + (j + 1) * (j + 1) * k2));
            g = (B * g2 + C * g1) / A;
            g1 = g2;
            g2 = g;
            if (fabs(g) > Chigh) {
                g *= Clow;
                g1 *= Clow;
                g2 *= Clow;
                Exponent += Ehigh;
            }
            else if (fabs(g) < Clow) {
                g *= Chigh;
                g1 *= Chigh;
                g2 *= Chigh;
                Exponent -= Ehigh;
            }
            //              printf("L,g,Exponent= %12i  %12.2e  %12.3f  \t\n",L,g,Exponent);
        }
    }
    else {
        for (int L = n - 3; L >= l; L--) {
            j = L + 1;
            A = 2. * n * sqrt((n * n - j * j) * (1. + (j - 1) * (j - 1) * k2));
            B = 4. * n * n - 4. * j * j + j * (2 * j + 1) * D;
            C = -2. * n * sqrt((n * n - (j + 1) * (j + 1)) * (1. + j * j * k2));
            g = (B * g4 + C * g3) / A;
            g3 = g4;
            g4 = g;
            if (fabs(g) > Chigh) {
                g *= Clow;
                g3 *= Clow;
                g4 *= Clow;
                Exponent += Ehigh;
            }
            else if (fabs(g) < Clow) {
                g *= Chigh;
                g3 *= Chigh;
                g4 *= Chigh;
                Exponent -= Ehigh;
            }
            //              printf("L,g,Exponent= %12i  %12.2e  %12.3f  \t\n",L,g,Exponent);
        }
    }

    g *= exp(Exponent);
    RadInt = n * n * g;

    // We have to multiply the output by 2/M_PI, due to the different wave func. normalizations;
    // Burgess uses a normalization of pi*delta(k^2-k'^2), and I use delta(E-E');
    // therefore my free wave function is sqrt(2./M_PI) times larger than the free wave func. of Burgess.

    return RadInt * RadInt * Cfac;
}

/////////////////////////////////////////////////////////////////////

double FBBRionization::SigmaPhotoionization(int n, int l, double omega)
{
    // Ionization energy of the (n,l) state in atomic units:
    double Enl = 1. / (2. * n * n);

    // Zero cross section below the ionization limit:
    if (omega < Enl)
        return 0.;

    // Outgoing electron energy in atomic units:
    double E = omega - Enl;

    double Sum = 0.;
    for (int sign = -1; sign <= +1; sign += 2) {
        int lp = l + sign;  // angular momentum quantum number of final state
        if (lp < 0)
            continue;

        double lplmax;
        if (sign < 0)
            lplmax = l;
        else
            lplmax = l + 1;

        double M2 = RadInt2BoundFreeBurgess(n, l, E, sign);
        Sum += lplmax * M2;
    }

    return fC * omega / (2. * l + 1.) * Sum;
}

/////////////////////////////////////////////////////////////////////

double FBBRionization::operator()(double E)
{
    // E is the energy of the outgoing electron in atomic units

    // Ionization energy of the (n,l) state in atomic units:
    double Enl = 1. / (2. * (double) (fn * fn));
    // Photon energy in atomic units:
    double omega = Enl + E;

    double omegakT = omega / (fAtomic_kB * fT);
    if (omegakT > 550.)
        omegakT = 550.;  // to avoid overflow
    double nbar = 1. / (exp(omegakT) - 1.);

    // Photon number density:
    double Nph = omega * omega / (M_PI * M_PI * fAtomic_C * fAtomic_C * fAtomic_C) * nbar;
    // Photoionization cross section:
    double sigma = SigmaPhotoionization(fn, fl, omega);

    return fAtomic_C * sigma * Nph;
}

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

RydbergCalculator::RydbergCalculator()
{
    for (int k = 1; k <= 2 * nMAX + 9; k++)
        LogN[k] = log((double) k);

    LogN[0] = 0.;
    LogNFactor[0] = 0;
    LogNFactor[1] = 0;

    for (int k = 2; k <= 2 * nMAX + 9; k++)
        LogNFactor[k] = LogNFactor[k - 1] + LogN[k];

    Ehigh = log(std::numeric_limits<double>::max()) * 0.25;
    Chigh = exp(Ehigh);
    Clow = exp(-Ehigh);

    fAtomic_C = 1. / katrin::KConst::Alpha();
    fAtomicTimeUnit = katrin::KConst::Hbar() / (katrin::KConst::Alpha() * katrin::KConst::Alpha() *
                                                katrin::KConst::M_el_kg() * katrin::KConst::C() * katrin::KConst::C());
    fAtomic_kB = 3.1668e-6;

    fFBBRIon = new FBBRionization(300., 1, 0);
};

/////////////////////////////////////////////////////////////////////

RydbergCalculator::~RydbergCalculator()
{
    delete fFBBRIon;
};

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::HypergeometricFHoangBinh(int a, int b, int c, double x, double& E)
{
    if (b > 0 || c <= 0) {
        intmsg(eError) << "Message from function HypergeometricFHoangBinh:" << ret << "b>0 or c<=0!" << ret
                       << "b,c= " << b << ", " << c << ret << "Program running is stopped !!! " << eom;
    }

    double A = 1.;
    double B = 1. - a / (double) c * x;

    if (b == 0) {
        E = 0;
        return A;
    }

    if (b == -1) {
        E = 0;
        return B;
    }

    E = 0;
    double F = 0., D;

    for (int n = -2; n >= b; n--) {
        D = 1. / (double) (n + 1 - c);
        F = D * ((n + 1) * (1. - x) * (B - A) + (n + 1 + a * x - c) * B);
        A = B;
        B = F;

        if (fabs(F) > Chigh) {
            F *= Clow;
            A *= Clow;
            B *= Clow;
            E += Ehigh;
        }
        else if (fabs(F) < Clow) {
            F *= Chigh;
            A *= Chigh;
            B *= Chigh;
            E -= Ehigh;
        }
    }

    return F;
}

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::RadInt2Gordon(int n, int l, int np, int sign)
{

    if (n < 1 || np < 1 || l < 0 || l > n - 1) {
        intmsg(eError) << "Message from function RadInt2Gordon:" << ret << "n and np have to be larger than 0!" << ret
                       << "l has to be larger than -1 and smaller than n !" << ret << "Program running is stopped !!! "
                       << eom;
    }

    if (n > nMAX || np > nMAX) {
        intmsg(eError) << "Message from function RadInt2Gordon" << ret << "n and np cannot be larger than nMAX!" << ret
                       << "Program running is stopped !!! " << eom;
    }

    if (sign != -1 && sign != +1) {
        intmsg(eError) << "Message from function RadInt2Gordon:" << ret << "sign has to be either -1 or +1!" << ret
                       << "Program running is stopped !!! " << eom;
    }

    int lp = l + sign;
    if (lp < 0 || lp > np - 1)
        return 0.;

    if (sign == 1) {
        int N, NP, L;
        N = np;
        NP = n;
        L = l + 1;
        n = N;
        np = NP;
        l = L;
    }

    if (n == np)  // exception!
    {
        return 9. / 4. * n * n * (n * n - l * l);
    }

    int nr = n - l - 1;
    int npr = np - l;

    // A, B, C, D, E0:
    double A = -LogN[4] - LogNFactor[2 * l - 1];
    double B = 0.5 * (LogNFactor[n + l] + LogNFactor[np + l - 1] - LogNFactor[n - l - 1] - LogNFactor[np - l]);
    double C = (l + 1) * (LogN[4] + LogN[n] + LogN[np]);
    double D = (n + np - 2 * l - 2) * LogN[abs(n - np)] - (n + np) * LogN[n + np];
    double E0 = A + B + C + D;

    // x, del2, sum2:
    double del2 = (n - np) * (n - np);
    double sum2 = (n + np) * (n + np);
    double x = -4. * n * np / del2;

    // F1, F2 :
    double E1, E2, F1, F2;

    if (n > np) {
        F1 = HypergeometricFHoangBinh(-nr, -npr, 2 * l, x, E1);
        F2 = HypergeometricFHoangBinh(-nr - 2, -npr, 2 * l, x, E2);
    }
    else {
        F1 = HypergeometricFHoangBinh(-npr, -nr, 2 * l, x, E1);
        F2 = HypergeometricFHoangBinh(-npr, -nr - 2, 2 * l, x, E2);
    }

    // RGordon:
    double RGordon = exp(E0 + E1) * F1 - (del2 / sum2) * exp(E0 + E2) * F2;


    return RGordon * RGordon;
}

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::Psp(int n, int l, int np, int sign)
{
    if (n == 1 || np >= n)
        return 0.;

    int lplmax;

    if (sign < 0)
        lplmax = l;
    else
        lplmax = l + 1;

    double C = 4. / (3. * fAtomic_C * fAtomic_C * fAtomic_C * (2. * l + 1.));
    double omega3 = std::pow(fabs(-1. / (2. * n * n) + 1. / (2. * np * np)), 3);

    return C * lplmax * omega3 * RadInt2Gordon(n, l, np, sign) / fAtomicTimeUnit;
}

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::Pspsum(int n, int l)
{
    if (n == 1)
        return 0.;

    double P = 0.;

    for (int np = 1; np < n; np++) {
        for (int sign = -1; sign <= +1; sign += 2) {
            P += Psp(n, l, np, sign);
        }
    }

    return P;
}

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::PBBR(double T, int n, int l, int np, int sign)
{

    int lplmax;

    if (sign < 0)
        lplmax = l;
    else
        lplmax = l + 1;

    double C = 4. / (3. * fAtomic_C * fAtomic_C * fAtomic_C * (2. * l + 1.));
    double omega = fabs(-1. / (2. * n * n) + 1. / (2. * np * np));
    double omega3 = std::pow(omega, 3);
    double omegakT = omega / (fAtomic_kB * T);

    if (omegakT > 550.)
        omegakT = 550.;

    double boltz = 1. / (exp(omegakT) - 1.);


    return boltz * C * lplmax * omega3 * RadInt2Gordon(n, l, np, sign) / fAtomicTimeUnit;
}

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::PBBRdecay(double T, int n, int l)
{
    double P = 0.;

    if (n == 1)
        return 0.;

    for (int np = 1; np < n; np++) {
        for (int sign = -1; sign <= +1; sign += 2) {
            P += PBBR(T, n, l, np, sign);
        }
    }

    return P;
}

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::PBBRexcitation(double T, int n, int l, int npmax)
{

    double P = 0.;

    for (int np = n + 1; np <= npmax; np++) {
        for (int sign = -1; sign <= +1; sign += 2) {
            P += PBBR(T, n, l, np, sign);
        }
    }

    return P;
}

/////////////////////////////////////////////////////////////////////

void RydbergCalculator::SpontaneousEmissionGenerator(int n, int l, double& Psptotal, int& np, int& lp)
{
    // Calculation of the discrete probability distributions function values PDF[i],  i=1,...,2*N.
    // PDF[i]= Psp(n,l,np,sign) / Psptotal;
    // Psp(n,l,np,sign): single spontaneous emission rate from initial state (n,l)
    // to final state (np,l+sign).
    // Psptotal:  sum of all single spontaneous emission rate values.
    // sign=-1: np goes from npmin to npmax, i from 1 to N;
    // sign=+1: np goes from npmin to npmax, i from N+1 to 2*N;

    if (n == 1)  // no decay from ground state!
    {
        Psptotal = 0.;
        np = 1;
        lp = 0;

        return;
    }

    if (n == 2 && l == 0)  // no 2s --> 1s decay !
    {
        Psptotal = 0.;
        np = 2;
        lp = 0;

        return;
    }

    int npmin = l;
    if (l == 0)
        npmin = 1;

    int npmax = n - 1;
    int N = npmax - npmin + 1;  // maximal value of the discrete random variable i

    //Code from here is identical to BBRTransitionGenerator

    double PDF[2 * nMAX + 2], CDF[2 * nMAX + 2];

    int sign;
    Psptotal = 0.;

    for (int i = 1; i <= 2 * N; i++) {
        if (i <= N) {
            sign = -1;
            np = npmin + i - 1;
        }
        else {
            sign = 1;
            np = npmin + i - N - 1;
        }

        PDF[i] = Psp(n, l, np, sign);
        Psptotal += PDF[i];
    }

    double tau = 1. / Psptotal;  // spontaneous emission lifetime

    // Normalized PDF values:
    for (int i = 1; i <= 2 * N; i++) {
        PDF[i] *= tau;
    }

    // Calculation of the discrete cumulative distribution function  CDF[i]:
    CDF[1] = 0.;
    for (int i = 2; i <= 2 * N + 1; i++) {
        CDF[i] = CDF[i - 1] + PDF[i - 1];
    }
    // CDF[2*N+1] has to be = 1 !

    // random number in [0,1]:
    double u = KRandom::GetInstance().Uniform(0., 1., true, false);

    // We search now the index value i so that  CDF[i] < u < CDF[i+1],
    // using the binary search algorithm:
    int i = 1;
    int j = 2 * N + 1;
    int k;
    do {
        k = (i + j) / 2;
        if (CDF[k] < u)
            i = k;
        else
            j = k;
    } while (j - i > 1);
    // Result of binary search:  i

    // Control:
    if (!(CDF[i] <= u && u <= CDF[i + 1]))
        printf("u,i,CDF[i],CDF[i+1]= %12.5f  %9i %12.5f   %12.5f  \t\n", u, i, CDF[i], CDF[i + 1]);

    // np and lp :
    if (i <= N) {
        sign = -1;
        np = npmin + i - 1;
    }
    else {
        sign = 1;
        np = npmin + i - N - 1;
    }

    lp = l + sign;

    return;
}

/////////////////////////////////////////////////////////////////////

void RydbergCalculator::BBRTransitionGenerator(double T, int n, int l, double& PBBRtotal, int& np, int& lp)
{
    // Calculation of the discrete probability distributions function values PDF[i],  i=1,...,2*N.
    // PDF[i]= PBBR(n,l,np,sign) / PBBRtotal;
    // PBBR(T,n,l,np,sign): single BBR induced transition rate from initial state (n,l)
    // to final state (np,l+sign).
    // PBBRtotal:  sum of all single BBR induced transition  rate values.
    // sign=-1: np goes from npmin to npmax, i from 1 to N;
    // sign=+1: np goes from npmin to npmax, i from N+1 to 2*N;


    int npmax = 10 * n;
    if (n > 80)
        npmax = 5 * n;

    if (npmax > nMAX)
        npmax = nMAX;

    //Code from here is identical to SpontaneousEmissionGenerator

    double PDF[2 * nMAX + 2], CDF[2 * nMAX + 2];

    int npmin = l;
    if (l == 0)
        npmin = 1;

    int N = npmax - npmin + 1;  // maximal value of the discrete random variable i

    int sign;
    PBBRtotal = 0.;

    for (int i = 1; i <= 2 * N; i++) {
        if (i <= N) {
            sign = -1;
            np = npmin + i - 1;
        }
        else {
            sign = 1;
            np = npmin + i - N - 1;
        }

        if (np != n)
            PDF[i] = PBBR(T, n, l, np, sign);
        else
            PDF[i] = 0.;

        PBBRtotal += PDF[i];
    }

    double tau = 1. / PBBRtotal;  // BBR induced transition lifetime

    // Normalized PDF values:
    for (int i = 1; i <= 2 * N; i++) {
        PDF[i] *= tau;
    }

    // Calculation of the discrete cumulative distribution function  CDF[i]:

    CDF[1] = 0.;
    for (int i = 2; i <= 2 * N + 1; i++) {
        CDF[i] = CDF[i - 1] + PDF[i - 1];
    }
    // CDF[2*N+1] has to be = 1 !

    // random number in [0,1]:
    double u = KRandom::GetInstance().Uniform(0., 1., true, false);

    // We search now the index value i so that  CDF[i] < u < CDF[i+1],
    // using the binary search algorithm:
    int i = 1;
    int j = 2 * N + 1;
    int k;
    do {
        k = (i + j) / 2;
        if (CDF[k] < u)
            i = k;
        else
            j = k;
    } while (j - i > 1);
    // Result of binary search:  i

    // Control:
    if (!(CDF[i] <= u && u <= CDF[i + 1]))
        printf("u,i,CDF[i],CDF[i+1]= %12.5f  %9i %12.5f   %12.5f  \t\n", u, i, CDF[i], CDF[i + 1]);

    // np and lp :
    if (i <= N) {
        sign = -1;
        np = npmin + i - 1;
    }
    else {
        sign = 1;
        np = npmin + i - N - 1;
    }

    lp = l + sign;

    return;
}

/////////////////////////////////////////////////////////////////////

double RydbergCalculator::PBBRionization(double T, int n, int l, double step1factor, double tol, int Ninteg)
{
    fFBBRIon->SetT(T);
    fFBBRIon->Setn(n);
    fFBBRIon->Setl(l);

    // Ionization energy of the (n,l) state in atomic units:
    double Enl = 1. / (2. * n * n);

    double step1 = Enl * step1factor;

    double xlimit = 0.01;

    double Pion = QuadGaussLegendre::IntegrateH(*fFBBRIon, step1, xlimit, tol, Ninteg);


    return Pion / fAtomicTimeUnit;
}


}  // namespace Kassiopeia

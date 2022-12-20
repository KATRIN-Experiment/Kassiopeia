#include "KSIntCalculatorHydrogen.h"

#include "KSInteractionsMessage.h"
#include "KSParticleFactory.h"
#include "KTextFile.h"
#include "KThreeVector.hh"
using katrin::KThreeVector;

#include "KConst.h"
#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{
/////////////////////////////////////
/////		Elastic	Base		/////
/////////////////////////////////////

KSIntCalculatorHydrogenElasticBase::KSIntCalculatorHydrogenElasticBase() = default;

KSIntCalculatorHydrogenElasticBase::~KSIntCalculatorHydrogenElasticBase() = default;

void KSIntCalculatorHydrogenElasticBase::CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection)
{
    CalculateCrossSection(aParticle.GetKineticEnergy_eV(), aCrossSection);
    return;
}

void KSIntCalculatorHydrogenElasticBase::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                            KSParticle& aFinalParticle, KSParticleQueue&)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    // outgoing primary

    double tLostKineticEnergy;
    double tTheta;
    double tPhi;

    CalculateTheta(tInitialKineticEnergy, tTheta);
    CalculateEloss(tInitialKineticEnergy, tTheta, tLostKineticEnergy);

    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetTime(anInitialParticle.GetTime());
    aFinalParticle.SetPosition(anInitialParticle.GetPosition());
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.SetLabel(GetName());

    fStepNInteractions = 1;
    fStepEnergyLoss = tLostKineticEnergy;
    fStepAngularChange = tTheta * 180. / katrin::KConst::Pi();

    return;
}

void KSIntCalculatorHydrogenElasticBase::CalculateTheta(const double anEnergy, double& aTheta)
{
    //double clight = 1. / katrin::KConst::Alpha();
    double T, c, b, G, a, gam, K2, Gmax;

    double tDiffCrossSection;
    double tRandom;

    if (anEnergy >= 250.)
        Gmax = 1.e-19;
    else if (anEnergy < 250. && anEnergy >= 150.)
        Gmax = 2.5e-19;
    else
        Gmax = 1.e-18;


    T = 0.5 * anEnergy / katrin::KConst::ERyd_eV();
    gam = 1. + T * katrin::KConst::Alpha() * katrin::KConst::Alpha();
    b = 2. / ((1. + gam) * T);

    while (true) {
        tRandom = KRandom::GetInstance().Uniform(0.0, 1.0, false, true);
        c = 1. + b - b * (2. + b) / (b + 2. * tRandom);
        K2 = 2. * T * (1. + gam) * fabs(1. - c);
        a = (4. + K2) * (4. + K2) / (gam * gam);
        CalculateDifferentialCrossSection(anEnergy, c, tDiffCrossSection);
        G = a * tDiffCrossSection;
        tRandom = KRandom::GetInstance().Uniform(0.0, 1.0, false, true);
        if (G > Gmax * tRandom)
            break;
    }

    aTheta = acos(c);
}

void KSIntCalculatorHydrogenElasticBase::CalculateDifferentialCrossSection(const double anEnergy, const double cosTheta,
                                                                           double& aCrossSection)
{
    // Nishimura et al., J. Phys. Soc. Jpn. 54 (1985) 1757.

    double a02 = katrin::KConst::BohrRadiusSquared();
    //double clight = 1. / katrin::KConst::Alpha(); // velocity of light in atomic units is 1/ alpha

    static double Cel[50] = {-0.512, -0.512, -0.509, -0.505, -0.499, -0.491, -0.476, -0.473, -0.462, -0.452,
                             -0.438, -0.422, -0.406, -0.388, -0.370, -0.352, -0.333, -0.314, -0.296, -0.277,
                             -0.258, -0.239, -0.221, -0.202, -0.185, -0.167, -0.151, -0.135, -0.120, -0.105,
                             -0.092, -0.070, -0.053, -0.039, -0.030, -0.024, -0.019, -0.016, -0.014, -0.013,
                             -0.012, -0.009, -0.008, -0.006, -0.005, -0.004, -0.003, -0.002, -0.002, -0.001};

    static double e[10] = {0., 3., 6., 12., 20., 32., 55., 85., 150., 250.};

    static double t[10] = {0., 10., 20., 30., 40., 60., 80., 100., 140., 180.};

    static double D[9][10] = {{2.9, 2.70, 2.5, 2.10, 1.80, 1.2000, 0.900, 1.0000, 1.600, 1.9},
                              {4.2, 3.60, 3.1, 2.50, 1.90, 1.1000, 0.800, 0.9000, 1.300, 1.4},
                              {6.0, 4.40, 3.2, 2.30, 1.80, 1.1000, 0.700, 0.5400, 0.500, 0.6},
                              {6.0, 4.10, 2.8, 1.90, 1.30, 0.6000, 0.300, 0.1700, 0.160, 0.23},
                              {4.9, 3.20, 2.0, 1.20, 0.80, 0.3000, 0.150, 0.0900, 0.050, 0.05},
                              {5.2, 2.50, 1.2, 0.64, 0.36, 0.1300, 0.050, 0.0300, 0.016, 0.02},
                              {4.0, 1.70, 0.7, 0.30, 0.16, 0.0500, 0.020, 0.0130, 0.010, 0.01},
                              {2.8, 1.10, 0.4, 0.15, 0.07, 0.0200, 0.010, 0.0070, 0.004, 0.003},
                              {1.2, 0.53, 0.2, 0.08, 0.03, 0.0074, 0.003, 0.0016, 0.001, 0.0008}};

    double T, K2, K, d, st1, st2, DH, gam, CelK, Ki, theta;
    aCrossSection = -1.0;

    T = 0.5 * anEnergy / katrin::KConst::ERyd_eV();
    if (anEnergy >= 250.) {
        gam = 1. + T * katrin::KConst::Alpha() * katrin::KConst::Alpha();  // relativistic correction factor
        K2 = 2. * T * (1. + gam) * (1. - cosTheta);
        if (K2 < 0.)
            K2 = 1.e-30;
        K = sqrt(K2);
        if (K < 1.e-9)
            K = 1.e-9;  // momentum transfer
        d = 1.4009;     // distance of protons in H2
        st1 = 8. + K2;
        st2 = 4. + K2;
        // DH is the diff. cross section for elastic electron scatt.
        // on atomic hydrogen within the first Born approximation :
        DH = 4. * st1 * st1 / (st2 * st2 * st2 * st2) * a02;
        // CelK calculation with linear interpolation.
        // CelK is the correction of the elastic electron
        // scatt. on molecular hydrogen compared to the independent atom
        // model.

        int i;

        if (K < 3.) {
            i = (int) (K / 0.1);  //WOLF int->double->int
            Ki = i * 0.1;
            CelK = Cel[i] + (K - Ki) / 0.1 * (Cel[i + 1] - Cel[i]);
        }
        else if (K >= 3. && K < 5.) {
            i = (int) (30 + (K - 3.) / 0.2);  //WOLF: int->double
            Ki = 3. + (i - 30) * 0.2;         //WOLF: int->double
            CelK = Cel[i] + (K - Ki) / 0.2 * (Cel[i + 1] - Cel[i]);
        }
        else if (K >= 5. && K < 9.49) {
            i = (int) (40 + (K - 5.) / 0.5);  //WOLF: int->double
            Ki = 5. + (i - 40) * 0.5;         //WOLF: int->double
            CelK = Cel[i] + (K - Ki) / 0.5 * (Cel[i + 1] - Cel[i]);
        }
        else
            CelK = 0.;

        aCrossSection = 2. * gam * gam * DH * (1. + sin(K * d) / (K * d)) * (1. + CelK);
    }  //end if anE>=250
    else {
        theta = acos(cosTheta) * 180. / katrin::KConst::Pi();
        for (int i = 0; i < 9; i++)
            if (anEnergy >= e[i] && anEnergy < e[i + 1])
                for (int j = 0; j < 9; j++)
                    if (theta >= t[j] && theta < t[j + 1])
                        aCrossSection =
                            1.e-20 * (D[i][j] + (D[i][j + 1] - D[i][j]) * (theta - t[j]) / (t[j + 1] - t[j]));
    }

    return;
}

/////////////////////////////////
/////		Elastic			/////
/////////////////////////////////

KSIntCalculatorHydrogenElastic::KSIntCalculatorHydrogenElastic() = default;

KSIntCalculatorHydrogenElastic::KSIntCalculatorHydrogenElastic(const KSIntCalculatorHydrogenElastic&) : KSComponent() {}

KSIntCalculatorHydrogenElastic* KSIntCalculatorHydrogenElastic::Clone() const
{
    return new KSIntCalculatorHydrogenElastic(*this);
}

KSIntCalculatorHydrogenElastic::~KSIntCalculatorHydrogenElastic() = default;

void KSIntCalculatorHydrogenElastic::CalculateCrossSection(const double anEnergie, double& aCrossSection)
{
    //        See: Liu, Phys. Rev. A35 (1987) 591,
    //        Trajmar, Phys Reports 97 (1983) 221.

    const double e[14] = {0., 1.5, 5., 7., 10., 15., 20., 30., 60., 100., 150., 200., 300., 400.};
    const double s[14] = {9.6, 13., 15., 12., 10., 7., 5.6, 3.3, 1.1, 0.9, 0.5, 0.36, 0.23, 0.15};

    const double emass = 1. / (katrin::KConst::Alpha() * katrin::KConst::Alpha());
    const double a02 = katrin::KConst::BohrRadiusSquared();

    double gam, T;
    T = anEnergie / (2 * katrin::KConst::ERyd_eV());
    if (anEnergie >= 400.) {
        gam = (emass + T) / emass;
        aCrossSection = gam * gam * katrin::KConst::Pi() / (2. * T) * (4.2106 - 1. / T) * a02;
    }
    else {
        for (unsigned int i = 0; i <= 12; i++) {
            if (anEnergie >= e[i] && anEnergie < e[i + 1])
                aCrossSection = 1.e-20 * (s[i] + (s[i + 1] - s[i]) * (anEnergie - e[i]) / (e[i + 1] - e[i]));
        }
    }

    return;
}

void KSIntCalculatorHydrogenElastic::CalculateEloss(const double anEnergie, const double aTheta, double& anEloss)
{
    double H2molmass = 69.e6;
    double emass = 1. / (katrin::KConst::Alpha() * katrin::KConst::Alpha());
    double cosTheta = cos(aTheta);

    anEloss = 2. * emass / H2molmass * (1. - cosTheta) * anEnergie;

    //check if electron won energy by elastic scattering on a molecule;
    //this keeps electron energies around the gas temperature

    if (anEnergie < 1.) {
        double rndNr = sqrt(-2. * log(KRandom::GetInstance().Uniform()));
        double rndAngle = 2. * katrin::KConst::Pi() * KRandom::GetInstance().Uniform();

        //generation of molecule velocity by maxwell-boltzmann distribution
        double Gx = rndNr * cos(rndAngle);
        double Gy = rndNr * sin(rndAngle);
        double Gz = sqrt(-2. * log(KRandom::GetInstance().Uniform())) *
                    cos(2. * katrin::KConst::Pi() * KRandom::GetInstance().Uniform());

        //thermal velocity of gas molecules
        double T = 300.;  //gas temperature
        double sigmaT = sqrt(katrin::KConst::kB() * T / (2. * katrin::KConst::M_prot_kg()));
        KThreeVector MolVelocity(sigmaT * Gx, sigmaT * Gy, sigmaT * Gz);

        //new electron velocity vector and energy:

        //assume electron velocity along z
        KThreeVector ElVelocity(0., 0., sqrt(2. * anEnergie * katrin::KConst::Q() / katrin::KConst::M_el_kg()));
        //relative velocity electron-molecule
        KThreeVector RelativeVelocity = ElVelocity - MolVelocity;
        //transformation into CMS
        KThreeVector CMSVelocity =
            (katrin::KConst::M_el_kg() / (katrin::KConst::M_el_kg() + katrin::KConst::M_prot_kg()) * ElVelocity +
             2. * katrin::KConst::M_prot_kg() * MolVelocity /
                 (katrin::KConst::M_el_kg() + katrin::KConst::M_prot_kg()));

        //generation of random direction
        KThreeVector Random(KRandom::GetInstance().Uniform(),
                            KRandom::GetInstance().Uniform(),
                            KRandom::GetInstance().Uniform());

        //new electron velocity
        ElVelocity = katrin::KConst::M_prot_kg() / (katrin::KConst::M_prot_kg() + katrin::KConst::M_el_kg()) *
                         RelativeVelocity.Magnitude() * Random.Unit() +
                     CMSVelocity;

        anEloss = anEnergie - katrin::KConst::M_el_kg() / (2. * katrin::KConst::Q()) * ElVelocity.Magnitude() *
                                  ElVelocity.Magnitude();
    }
    return;
}


/////////////////////////////////
/////		Vibration		/////
/////////////////////////////////

KSIntCalculatorHydrogenVib::KSIntCalculatorHydrogenVib() = default;

KSIntCalculatorHydrogenVib::KSIntCalculatorHydrogenVib(const KSIntCalculatorHydrogenVib&) : KSComponent() {}

KSIntCalculatorHydrogenVib* KSIntCalculatorHydrogenVib::Clone() const
{
    return new KSIntCalculatorHydrogenVib(*this);
}

KSIntCalculatorHydrogenVib::~KSIntCalculatorHydrogenVib() = default;

void KSIntCalculatorHydrogenVib::CalculateCrossSection(const double anEnergie, double& aCrossSection)
{
    unsigned int i;

    static double sigma1[8] = {0.0, 0.006, 0.016, 0.027, 0.033, 0.045, 0.057, 0.065};

    static double sigma2[9] = {0.065, 0.16, 0.30, 0.36, 0.44, 0.47, 0.44, 0.39, 0.34};

    static double sigma3[7] = {0.34, 0.27, 0.21, 0.15, 0.12, 0.08, 0.07};

    if (anEnergie <= 0.5 || anEnergie > 10.) {
        aCrossSection = 0.;
    }
    else {
        if (anEnergie >= 0.5 && anEnergie < 1.0) {
            i = (anEnergie - 0.5) / 0.1;
            aCrossSection = 1.e-20 * (sigma1[i] + (sigma1[i + 1] - sigma1[i]) * (anEnergie - 0.5 - i * 0.1) * 10.);
        }
        else {
            if (anEnergie >= 1.0 && anEnergie < 5.0) {
                i = (anEnergie - 1.0) / 0.5;
                aCrossSection = 1.e-20 * (sigma2[i] + (sigma2[i + 1] - sigma2[i]) * (anEnergie - 1.0 - i * 0.5) * 2.);
            }
            else {
                i = (anEnergie - 5.0) / 1.0;
                aCrossSection = 1.e-20 * (sigma3[i] + (sigma3[i + 1] - sigma3[i]) * (anEnergie - 5.0 - i * 1.0));
            }
        }
    }
    return;
}

void KSIntCalculatorHydrogenVib::CalculateEloss(const double, const double, double& anEloss)
{
    anEloss = 0.5;
}

/////////////////////////////////
/////		Rot02			/////
/////////////////////////////////

KSIntCalculatorHydrogenRot02::KSIntCalculatorHydrogenRot02() = default;

KSIntCalculatorHydrogenRot02::KSIntCalculatorHydrogenRot02(const KSIntCalculatorHydrogenRot02&) : KSComponent() {}

KSIntCalculatorHydrogenRot02* KSIntCalculatorHydrogenRot02::Clone() const
{
    return new KSIntCalculatorHydrogenRot02(*this);
}

KSIntCalculatorHydrogenRot02::~KSIntCalculatorHydrogenRot02() = default;

void KSIntCalculatorHydrogenRot02::CalculateCrossSection(const double anEnergie, double& aCrossSection)
{
    unsigned int i;

    static double sigma2[8] = {0.065, 0.069, 0.073, 0.077, 0.081, 0.085, 0.088, 0.090};

    static double sigma3[10] = {0.09, 0.11, 0.15, 0.20, 0.26, 0.32, 0.39, 0.47, 0.55, 0.64};

    static double sigma4[9] = {0.64, 1.04, 1.37, 1.58, 1.70, 1.75, 1.76, 1.73, 1.69};

    static double sigma5[7] = {1.69, 1.58, 1.46, 1.35, 1.25, 1.16, 1.0};

    static double DeltaE = 0.045;

    if (anEnergie <= DeltaE + 1.e-8 || anEnergie > 10.) {
        aCrossSection = 0.;
    }
    else {
        if (anEnergie >= 0.045 && anEnergie < 0.1) {
            i = (anEnergie - 0.045) / 0.01;
            aCrossSection = 1.e-20 * (sigma2[i] + (sigma2[i + 1] - sigma2[i]) * (anEnergie - 0.045 - i * 0.01) * 100.);
        }
        else {
            if (anEnergie >= 0.1 && anEnergie < 1.0) {
                i = (anEnergie - 0.1) / 0.1;
                aCrossSection = 1.e-20 * (sigma3[i] + (sigma3[i + 1] - sigma3[i]) * (anEnergie - 0.1 - i * 0.1) * 10.);
            }
            else {
                if (anEnergie >= 1.0 && anEnergie < 5.0) {
                    i = (anEnergie - 1.0) / 0.5;
                    aCrossSection =
                        1.e-20 * (sigma4[i] + (sigma4[i + 1] - sigma4[i]) * (anEnergie - 1.0 - i * 0.5) * 2.);
                }
                else {
                    i = (anEnergie - 5.0) / 1.0;
                    aCrossSection = 1.e-20 * (sigma5[i] + (sigma5[i + 1] - sigma5[i]) * (anEnergie - 5.0 - i * 1.0));
                }
            }
        }
    }

    return;
}

void KSIntCalculatorHydrogenRot02::CalculateEloss(const double, const double, double& anEloss)
{
    anEloss = 0.045;
}

/////////////////////////////////
/////		Rot13			/////
/////////////////////////////////

KSIntCalculatorHydrogenRot13::KSIntCalculatorHydrogenRot13() = default;

KSIntCalculatorHydrogenRot13::KSIntCalculatorHydrogenRot13(const KSIntCalculatorHydrogenRot13&) : KSComponent() {}

KSIntCalculatorHydrogenRot13* KSIntCalculatorHydrogenRot13::Clone() const
{
    return new KSIntCalculatorHydrogenRot13(*this);
}

KSIntCalculatorHydrogenRot13::~KSIntCalculatorHydrogenRot13() = default;

void KSIntCalculatorHydrogenRot13::CalculateCrossSection(const double anEnergie, double& aCrossSection)
{
    unsigned int i;

    static double sigma2[6] = {0.035, 0.038, 0.041, 0.044, 0.047, 0.05};

    static double sigma3[10] = {0.05, 0.065, 0.09, 0.11, 0.14, 0.18, 0.21, 0.25, 0.29, 0.33};

    static double sigma4[9] = {0.33, 0.55, 0.79, 0.94, 1.01, 1.05, 1.05, 1.04, 1.01};

    static double sigma5[7] = {1.01, 0.95, 0.88, 0.81, 0.75, 0.69, 0.62};

    static double DeltaE = 0.075;

    if (anEnergie <= DeltaE + 1.e-8 || anEnergie > 10.) {
        aCrossSection = 0.;
    }
    else {
        if (anEnergie >= 0.075 && anEnergie < 0.1) {
            i = (anEnergie - 0.075) / 0.01;
            aCrossSection = 1.e-20 * (sigma2[i] + (sigma2[i + 1] - sigma2[i]) * (anEnergie - 0.075 - i * 0.01) * 100.);
        }
        else {
            if (anEnergie >= 0.1 && anEnergie < 1.0) {
                i = (anEnergie - 0.1) / 0.1;
                aCrossSection = 1.e-20 * (sigma3[i] + (sigma3[i + 1] - sigma3[i]) * (anEnergie - 0.1 - i * 0.1) * 10.);
            }
            else {
                if (anEnergie >= 1.0 && anEnergie < 5.0) {
                    i = (anEnergie - 1.0) / 0.5;
                    aCrossSection =
                        1.e-20 * (sigma4[i] + (sigma4[i + 1] - sigma4[i]) * (anEnergie - 1.0 - i * 0.5) * 2.);
                }
                else {
                    i = (anEnergie - 5.0) / 1.0;
                    aCrossSection = 1.e-20 * (sigma5[i] + (sigma5[i + 1] - sigma5[i]) * (anEnergie - 5.0 - i * 1.0));
                }
            }
        }
    }
    return;
}

void KSIntCalculatorHydrogenRot13::CalculateEloss(const double, const double, double& anEloss)
{
    anEloss = 0.075;
}

/////////////////////////////////
/////		Rot20			/////
/////////////////////////////////

KSIntCalculatorHydrogenRot20::KSIntCalculatorHydrogenRot20() = default;

KSIntCalculatorHydrogenRot20::KSIntCalculatorHydrogenRot20(const KSIntCalculatorHydrogenRot20&) : KSComponent() {}

KSIntCalculatorHydrogenRot20* KSIntCalculatorHydrogenRot20::Clone() const
{
    return new KSIntCalculatorHydrogenRot20(*this);
}

KSIntCalculatorHydrogenRot20::~KSIntCalculatorHydrogenRot20() = default;

void KSIntCalculatorHydrogenRot20::CalculateCrossSection(const double anEnergie, double& aCrossSection)
{
    unsigned int i;

    double Ep = anEnergie + 0.045;

    static double sigma2[8] = {0.065, 0.069, 0.073, 0.077, 0.081, 0.085, 0.088, 0.090};

    static double sigma3[10] = {0.09, 0.11, 0.15, 0.20, 0.26, 0.32, 0.39, 0.47, 0.55, 0.64};

    static double sigma4[9] = {0.64, 1.04, 1.37, 1.58, 1.70, 1.75, 1.76, 1.73, 1.69};

    static double sigma5[7] = {1.69, 1.58, 1.46, 1.35, 1.25, 1.16, 1.0};

    static double DeltaE = 0.045;

    if (Ep <= DeltaE + 1.e-8 || Ep > 10.) {
        aCrossSection = 0.;
    }
    else {
        if (Ep >= 0.045 && Ep < 0.1) {
            i = (Ep - 0.045) / 0.01;
            aCrossSection = 1.e-20 * (sigma2[i] + (sigma2[i + 1] - sigma2[i]) * (Ep - 0.045 - i * 0.01) * 100.);
        }
        else {
            if (Ep >= 0.1 && Ep < 1.0) {
                i = (Ep - 0.1) / 0.1;
                aCrossSection = 1.e-20 * (sigma3[i] + (sigma3[i + 1] - sigma3[i]) * (Ep - 0.1 - i * 0.1) * 10.);
            }
            else {
                if (Ep >= 1.0 && Ep < 5.0) {
                    i = (Ep - 1.0) / 0.5;
                    aCrossSection = 1.e-20 * (sigma4[i] + (sigma4[i + 1] - sigma4[i]) * (Ep - 1.0 - i * 0.5) * 2.);
                }
                else {
                    i = (Ep - 5.0) / 1.0;
                    aCrossSection = 1.e-20 * (sigma5[i] + (sigma5[i + 1] - sigma5[i]) * (Ep - 5.0 - i * 1.0));
                }
            }
        }
    }

    aCrossSection = 1. / 5. * Ep / anEnergie * aCrossSection;
}

void KSIntCalculatorHydrogenRot20::CalculateEloss(const double, const double, double& anEloss)
{
    anEloss = -0.045;
}

/////////////////////////////////////
/////		Excitation Base		/////
/////////////////////////////////////

KSIntCalculatorHydrogenExcitationBase::KSIntCalculatorHydrogenExcitationBase() :
    T(20000. / (2 * katrin::KConst::ERyd_eV())),
    Ecen(12.6 / 27.21)
{
    initialize_sum();
}

void KSIntCalculatorHydrogenExcitationBase::initialize_sum()
{
    xmin = Ecen * Ecen / (2. * T);
    ymin = log(xmin);
    ymax = log(8. * T + xmin);
    dy = (ymax - ymin) / 1000.;

    // Initialization of the sum[] vector, and fmax calculation:
    fmax = 0;
    for (int n = 0; n <= 1000; n++) {
        y = ymin + dy * n;
        K = exp(y / 2.);

        sum[n] = sumexc(K);

        if (sum[n] > fmax)
            fmax = sum[n];
    }

    fmax = 1.05 * fmax;
}

KSIntCalculatorHydrogenExcitationBase::~KSIntCalculatorHydrogenExcitationBase() = default;

void KSIntCalculatorHydrogenExcitationBase::CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection)
{
    CalculateCrossSection(aParticle.GetKineticEnergy_eV(), aCrossSection);
    return;
}

void KSIntCalculatorHydrogenExcitationBase::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                               KSParticle& aFinalParticle, KSParticleQueue&)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    // outgoing primary

    double tLostKineticEnergy;
    double tTheta;
    double tPhi;

    CalculateTheta(tInitialKineticEnergy, tTheta);
    CalculateEloss(tInitialKineticEnergy, tTheta, tLostKineticEnergy);

    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetTime(anInitialParticle.GetTime());
    aFinalParticle.SetPosition(anInitialParticle.GetPosition());
    aFinalParticle.SetMomentum(tFinalDirection);
    aFinalParticle.SetKineticEnergy_eV(tInitialKineticEnergy - tLostKineticEnergy);
    aFinalParticle.SetLabel(GetName());

    fStepNInteractions = 1;
    fStepEnergyLoss = tLostKineticEnergy;

    return;
}

void KSIntCalculatorHydrogenExcitationBase::CalculateTheta(const double anEnergy, double& aTheta)
{
    double anEnergy_HartreeFock = anEnergy / (2 * katrin::KConst::ERyd_eV());

    if (anEnergy >= 100.) {
        xmin = Ecen * Ecen / (2. * anEnergy_HartreeFock);
        ymin = log(xmin);
        ymax = log(8. * anEnergy_HartreeFock + xmin);
        dy = (ymax - ymin) / 1000.;

        // Generation of y values with the Neumann acceptance-rejection method:
        while (true) {
            y = KRandom::GetInstance().Uniform(ymin, ymax, false, true);
            K = exp(y / 2.);
            fy = sumexc(K);
            if (KRandom::GetInstance().Uniform(0.0, fmax, false, true) < fy)
                break;
        }

        // Calculation of c = cos(Theta) and Theta:
        x = exp(y);
        c = 1. - (x - xmin) / (4. * anEnergy_HartreeFock);
    }
    else {
        if (anEnergy <= 25.)
            Dmax = 60.;
        else if (anEnergy > 25. && anEnergy <= 35.)
            Dmax = 95.;
        else if (anEnergy > 35. && anEnergy <= 50.)
            Dmax = 150.;
        else
            Dmax = 400.;

        while (true) {
            c = KRandom::GetInstance().Uniform(-1.0, 1.0, false, true);
            CalculateDifferentialCrossSection(anEnergy, c, D);
            D *= 1.e22;  //this is important

            if (KRandom::GetInstance().Uniform(0.0, Dmax, false, true) < D)
                break;
        }
    }

    aTheta = acos(c);
}

void KSIntCalculatorHydrogenExcitationBase::CalculateDifferentialCrossSection(const double anEnergy,
                                                                              const double cosTheta,
                                                                              double& aCrossSection)
{
    double K2, K, T, theta;
    aCrossSection = 0.;
    double a02 = katrin::KConst::BohrRadiusSquared();

    static double EE = 12.6 / (2 * katrin::KConst::ERyd_eV());
    static double e[5] = {0., 25., 35., 50., 100.};
    static double t[9] = {0., 10., 20., 30., 40., 60., 80., 100., 180.};
    static double D[4][9] = {{60., 43., 27., 18., 13., 8., 6., 6., 6.},
                             {
                                 95.,
                                 70.,
                                 21.,
                                 9.,
                                 6.,
                                 3.,
                                 2.,
                                 2.,
                                 2.,
                             },
                             {150., 120., 32., 8., 3.7, 1.9, 1.2, 0.8, 0.8},
                             {400., 200., 12., 2., 1.4, 0.7, 0.3, 0.2, 0.2}};

    T = anEnergy / (2 * katrin::KConst::ERyd_eV());

    if (anEnergy >= 100.) {
        //squared momentum transfer calculated acc. to http://dx.doi.org/10.1080/00268978000103701 page 1504
        K2 = 4. * T * (1. - EE / (2. * T) - sqrt(1. - EE / T) * cosTheta);

        if (K2 < 1.e-9)
            K2 = 1.e-9;

        K = sqrt(K2);                               // momentum transfer
        aCrossSection = 2. / K2 * sumexc(K) * a02;  //formula (13) in http://dx.doi.org/10.1080/00268978000103701
    }
    else if (anEnergy <= 10.)
        aCrossSection = 0.;
    else {
        theta = acos(cosTheta) * 180. / katrin::KConst::Pi();

        for (int i = 0; i < 4; i++) {
            if (anEnergy >= e[i] && anEnergy < e[i + 1]) {
                for (int j = 0; j < 8; j++) {
                    if (theta >= t[j] && theta < t[j + 1])
                        aCrossSection =
                            1.e-22 * (D[i][j] + (D[i][j + 1] - D[i][j]) * (theta - t[j]) / (t[j + 1] - t[j]));
                }
            }
        }
    }

    return;
}

double KSIntCalculatorHydrogenExcitationBase::sumexc(double K)
{
    // exchanged momentum in values of hbar/a0
    static double Kvec[15] = {0., 0.1, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.5, 1.8, 2., 2.5, 3., 4., 5.};

    // rotationally averaged, generalized oscillator strengths taken from
    // Arrighini, G.P. and Biondi, F. and Guidotti, C
    // "A study of the inelastic scattering of fast electrons from molecular hydrogen"
    // J. Mol. Phys. 41 1501-1514 1980
    // http://dx.doi.org/10.1080/00268978000103701
    // these values correspond, for each Excitation to the momentum values in Kvec
    static double fvec[7][15] = {{2.907e-1,
                                  2.845e-1,
                                  2.665e-1,
                                  2.072e-1,
                                  1.389e-1,  // B
                                  8.238e-2,
                                  4.454e-2,
                                  2.269e-2,
                                  7.789e-3,
                                  2.619e-3,
                                  1.273e-3,
                                  2.218e-4,
                                  4.372e-5,
                                  2.889e-6,
                                  4.247e-7},
                                 {3.492e-1,
                                  3.367e-1,
                                  3.124e-1,
                                  2.351e-1,
                                  1.507e-1,  // C
                                  8.406e-2,
                                  4.214e-2,
                                  1.966e-2,
                                  5.799e-3,
                                  1.632e-3,
                                  6.929e-4,
                                  8.082e-5,
                                  9.574e-6,
                                  1.526e-7,
                                  7.058e-9},
                                 {6.112e-2,
                                  5.945e-2,
                                  5.830e-2,
                                  5.072e-2,
                                  3.821e-2,  // Bp
                                  2.579e-2,
                                  1.567e-2,
                                  8.737e-3,
                                  3.305e-3,
                                  1.191e-3,
                                  6.011e-4,
                                  1.132e-4,
                                  2.362e-5,
                                  1.603e-6,
                                  2.215e-7},
                                 {2.066e-2,
                                  2.127e-2,
                                  2.137e-2,
                                  1.928e-2,
                                  1.552e-2,  // Bpp
                                  1.108e-2,
                                  7.058e-3,
                                  4.069e-3,
                                  1.590e-3,
                                  5.900e-4,
                                  3.046e-4,
                                  6.142e-5,
                                  1.369e-5,
                                  9.650e-7,
                                  1.244e-7},
                                 {9.405e-2,
                                  9.049e-2,
                                  8.613e-2,
                                  7.301e-2,
                                  5.144e-2,  // D
                                  3.201e-2,
                                  1.775e-2,
                                  8.952e-3,
                                  2.855e-3,
                                  8.429e-4,
                                  3.655e-4,
                                  4.389e-5,
                                  5.252e-6,
                                  9.010e-8,
                                  7.130e-9},
                                 {4.273e-2,
                                  3.862e-2,
                                  3.985e-2,
                                  3.362e-2,
                                  2.486e-2,  // Dp
                                  1.612e-2,
                                  9.309e-3,
                                  4.856e-3,
                                  1.602e-3,
                                  4.811e-4,
                                  2.096e-4,
                                  2.498e-5,
                                  2.905e-6,
                                  5.077e-8,
                                  6.583e-9},
                                 {0.000e-3,
                                  2.042e-3,
                                  7.439e-3,
                                  2.200e-2,
                                  3.164e-2,  // EF
                                  3.161e-2,
                                  2.486e-2,
                                  1.664e-2,
                                  7.562e-3,
                                  3.044e-3,
                                  1.608e-3,
                                  3.225e-4,
                                  7.120e-5,
                                  6.290e-6,
                                  1.066e-6}};

    // energies of the exicted states
    static double EeV[7] = {12.73, 13.20, 14.77, 15.3, 14.93, 15.4, 13.06};

    int jmin = 0;
    int nnmax = 6;
    double En, f[7], x4[4], f4[4], total_sum;

    total_sum = 0.;
    for (int n = 0; n <= nnmax; n++) {
        En = EeV[n] / (2 * katrin::KConst::ERyd_eV());  // En is the excitation energy in Hartree atomic units

        if (K >= 5.)
            f[n] = 0.;  //not in table, and small
        else if (K >= 4. && K < 5.)
            f[n] = fvec[n][13] + (K - 4.) * (fvec[n][14] - fvec[n][13]);  //linear interpolation
        else if (K >= 3. && K < 4.)
            f[n] = fvec[n][12] + (K - 3.) * (fvec[n][13] - fvec[n][12]);  //linear interpolation
        else {
            for (int j = 0; j < 14; j++) {
                if (K >= Kvec[j] && K <= Kvec[j + 1])
                    jmin = j - 1;
            }

            if (jmin < 0)
                jmin = 0;
            if (jmin > 11)
                jmin = 11;

            for (int j = 0; j <= 3; j++) {
                x4[j] = Kvec[jmin + j];     //points for lagrange interpolation
                f4[j] = fvec[n][jmin + j];  //are selected here - jmin and the next 3 points < sounds dull
            }

            f[n] = Lagrange(4, x4, f4, K);  // finally fn is interpolated
        }
        total_sum += f[n] / En;  //hm sum is over f/E, check how this fits with the formula
    }
    return total_sum;
}

double KSIntCalculatorHydrogenExcitationBase::Lagrange(int n, double* xn, double* fn, double x)
{
    //lagrange interpolation.
    //xn Stuetzstellen
    //fn Funktionswerte
    //n < 100   <= lol

    double f, a[100], b[100], aa, bb;
    f = 0.;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            a[i] = x - xn[i];
            b[i] = xn[j] - xn[i];
        }
        a[j] = b[j] = aa = bb = 1.;

        for (int i = 0; i < n; i++) {
            aa = aa * a[i];
            bb = bb * b[i];
        }
        f += fn[j] * aa / bb;
    }
    return f;
}

/////////////////////////////////
/////		Excitation B	/////
/////////////////////////////////

KSIntCalculatorHydrogenExcitationB::KSIntCalculatorHydrogenExcitationB() : Emin(12.5)
{
    initialize_sum();
    FrankCondonSum = 0.;
    for (double FrankCondonFactor : FrankCondonFactors)
        FrankCondonSum += FrankCondonFactor;
    ElExcitationFCSum = 0.;
    for (double i : FCFactorsElectronicExcitation)
        ElExcitationFCSum += i;
}

KSIntCalculatorHydrogenExcitationB::KSIntCalculatorHydrogenExcitationB(
    const KSIntCalculatorHydrogenExcitationB& aCopy) :
    KSComponent(aCopy),
    Emin(aCopy.Emin)
{
    initialize_sum();
    FrankCondonSum = 0.;
    for (int i = 0; i < 14; i++)
        FrankCondonSum += FrankCondonFactors[i];
    ElExcitationFCSum = 0.;
    for (double i : FCFactorsElectronicExcitation)
        ElExcitationFCSum += i;
}

KSIntCalculatorHydrogenExcitationB* KSIntCalculatorHydrogenExcitationB::Clone() const
{
    return new KSIntCalculatorHydrogenExcitationB(*this);
}

KSIntCalculatorHydrogenExcitationB::~KSIntCalculatorHydrogenExcitationB() = default;

// Energy values of the B vibrational states:
//   (from: Phys. Rev. A51 (1995) 3745 , in Hartree atomic units)
const double KSIntCalculatorHydrogenExcitationB ::EnergyLevels[28] = {
    0.411, 0.417, 0.423, 0.428, 0.434, 0.439, 0.444, 0.449, 0.454, 0.459, 0.464, 0.468, 0.473, 0.477,
    0.481, 0.485, 0.489, 0.493, 0.496, 0.500, 0.503, 0.507, 0.510, 0.513, 0.516, 0.519, 0.521, 0.524};

// Franck-Condon factors of the B vibrational states:
//   (from: Phys. Rev. A51 (1995) 3745 )
const double KSIntCalculatorHydrogenExcitationB ::FrankCondonFactors[28] = {
    4.2e-3, 1.5e-2, 3.0e-2, 4.7e-2, 6.3e-2, 7.3e-2, 7.9e-2, 8.0e-2, 7.8e-2, 7.3e-2, 6.6e-2, 5.8e-2, 5.1e-2, 4.4e-2,
    3.7e-2, 3.1e-2, 2.6e-2, 2.2e-2, 1.8e-2, 1.5e-2, 1.3e-2, 1.1e-2, 8.9e-3, 7.4e-3, 6.2e-3, 5.2e-3, 4.3e-3, 3.6e-3};

// Parameters for a logarithmic expansion of the Crosssection which were originally obtained from
// a fit to experimental data.
// Taken from R.K Janev, W.D.Langer, K.Evans Jr., D.E.Post Jr. "Elementary Processes in Hydrogen-Helium-Plasmas"
// Springer (1987)
// ISBN 3-540-17588-1 Springer-Verlag Berlin Heidelberg New York
// ISBN 0-387-17588-1 Springer-Verlag New York Berlin Heidelberg
// page 38. Process 2.2.2
const double KSIntCalculatorHydrogenExcitationB ::CrossSectionParameters[9] = {-4.2935194e2,
                                                                               5.1122109e2,
                                                                               -2.8481279e2,
                                                                               8.8310338e1,
                                                                               -1.6659591e1,
                                                                               1.9579609,
                                                                               -1.4012824e-1,
                                                                               5.5911348e-3,
                                                                               -9.5370103e-5};

const double KSIntCalculatorHydrogenExcitationB ::FCFactorsElectronicExcitation[7] =
    {35.86, 40.05, 6.58, 2.26, 9.61, 4.08, 1.54};

void KSIntCalculatorHydrogenExcitationB::CalculateCrossSection(const double anEnergy, double& aCrossSection)
{
    //********
    // B state
    //********


    double lnsigma, lnE, lnEn;

    lnE = log(anEnergy);

    lnEn = 1.;
    lnsigma = 0.;

    if (anEnergy < Emin)
        aCrossSection = 0.;
    else if (anEnergy < 226.36976843987776)  // Energy limit is the numerically computed intersection
    {                                        // betweeen the two cases
        for (double CrossSectionParameter : CrossSectionParameters) {
            lnsigma += CrossSectionParameter * lnEn;
            lnEn = lnEn * lnE;
        }
        aCrossSection = 1.e-4 * exp(lnsigma);
    }
    else {
        aCrossSection = 4 * katrin::KConst::Pi() * katrin::KConst::BohrRadiusSquared() * katrin::KConst::ERyd_eV() /
                        anEnergy * (0.8 * log(anEnergy / katrin::KConst::ERyd_eV()) + 0.28) *
                        FCFactorsElectronicExcitation[0] / ElExcitationFCSum;
    }
}

void KSIntCalculatorHydrogenExcitationB::CalculateEloss(const double, const double, double& anEloss)
{
    // B state; we generate now a vibrational state
    // using the Frank-Condon factors

    int nBVibrationalStates = 28;  // the number of B vibrational states in our calculation

    double tRandom = KRandom::GetInstance().Uniform(0., FrankCondonSum, false, true);

    int v;
    for (v = 0; v < nBVibrationalStates; v++) {
        tRandom -= FrankCondonFactors[v];
        if (tRandom < 0)
            break;
    }

    anEloss = EnergyLevels[v] * 2 * katrin::KConst::ERyd_eV();
}

/////////////////////////////////
/////		Excitation C	/////
/////////////////////////////////

KSIntCalculatorHydrogenExcitationC::KSIntCalculatorHydrogenExcitationC() : Emin(15.8)
{
    initialize_sum();
    FrankCondonSum = 0;
    for (double FrankCondonFactor : FrankCondonFactors)
        FrankCondonSum += FrankCondonFactor;
}

// Energy values of the C vibrational states:
//   (from: Phys. Rev. A51 (1995) 3745 , in Hartree atomic units)
const double KSIntCalculatorHydrogenExcitationC ::EnergyLevels[14] =
    {0.452, 0.462, 0.472, 0.481, 0.490, 0.498, 0.506, 0.513, 0.519, 0.525, 0.530, 0.534, 0.537, 0.539};

// Franck-Condon factors of the C vibrational states:
//   (from: Phys. Rev. A51 (1995) 3745 )
const double KSIntCalculatorHydrogenExcitationC ::FrankCondonFactors[14] =
    {1.2e-1, 1.9e-1, 1.9e-1, 1.5e-1, 1.1e-1, 7.5e-2, 5.0e-2, 3.3e-2, 2.2e-2, 1.4e-2, 9.3e-3, 6.0e-3, 3.7e-3, 1.8e-3};


// Parameters for a logarithmic expansion of the Crosssection which were originally obtained from
// a fit to experimental data.
// Taken from R.K Janev, W.D.Langer, K.Evans Jr., D.E.Post Jr. "Elementary Processes in Hydrogen-Helium-Plasmas"
// Springer (1987)
// ISBN 3-540-17588-1 Springer-Verlag Berlin Heidelberg New York
// ISBN 0-387-17588-1 Springer-Verlag New York Berlin Heidelberg
// page 40. Process 2.2.3
const double KSIntCalculatorHydrogenExcitationC ::CrossSectionParameters[9] = {-8.1942684e2,
                                                                               9.8705099e2,
                                                                               -5.3095543e2,
                                                                               1.5917023e2,
                                                                               -2.9121036e1,
                                                                               3.3321027,
                                                                               -2.3305961e-1,
                                                                               9.1191781e-3,
                                                                               -1.5298950e-4};

KSIntCalculatorHydrogenExcitationC::KSIntCalculatorHydrogenExcitationC(
    const KSIntCalculatorHydrogenExcitationC& aCopy) :
    KSComponent(aCopy),
    Emin(aCopy.Emin)
{
    initialize_sum();
    FrankCondonSum = 0;
    for (double FrankCondonFactor : FrankCondonFactors)
        FrankCondonSum += FrankCondonFactor;
}

KSIntCalculatorHydrogenExcitationC* KSIntCalculatorHydrogenExcitationC::Clone() const
{
    return new KSIntCalculatorHydrogenExcitationC(*this);
}

KSIntCalculatorHydrogenExcitationC::~KSIntCalculatorHydrogenExcitationC() = default;

void KSIntCalculatorHydrogenExcitationC::CalculateCrossSection(const double anEnergy, double& aCrossSection)
{
    //*********
    // C state:
    //*********

    if (anEnergy < 383.5255342162897)  // Energy limit is the numerically computed intersection
    {                                  // betweeen the two cases
        double lnsigma, lnE, lnEn;

        lnE = log(anEnergy);

        lnEn = 1.;
        lnsigma = 0.;

        if (anEnergy < Emin)
            aCrossSection = 0.;
        else {
            for (double CrossSectionParameter : CrossSectionParameters) {
                lnsigma += CrossSectionParameter * lnEn;
                lnEn = lnEn * lnE;
            }
            aCrossSection = 1.e-4 * exp(lnsigma);
        }
    }
    else {
        // Born-Fit given in "Elementary Processes in Hydrogen-Helium-Plasmas" parameters: p. 240 formula: p.234
        aCrossSection = 1.e-4 * 1.15646e-16 * std::pow((11.7 / anEnergy), (1.00587e+0)) * log(anEnergy / 11.70);
    }
}

void KSIntCalculatorHydrogenExcitationC::CalculateEloss(const double, const double, double& anEloss)
{
    // C state; we generate now a vibrational state,
    // using the Franck-Condon factors

    int nCVibrationalStates = 14;  // the number of C vibrational states in our calculation

    double tRandom = KRandom::GetInstance().Uniform(0., FrankCondonSum, false, true);

    int v;
    for (v = 0; v < nCVibrationalStates; v++) {
        tRandom -= FrankCondonFactors[v];
        if (tRandom < 0)
            break;
    }


    anEloss = EnergyLevels[v] * 2 * katrin::KConst::ERyd_eV();
}

/////////////////////////////////
/////   Dissoziation 10 eV	/////
/////////////////////////////////

KSIntCalculatorHydrogenDissoziation10::KSIntCalculatorHydrogenDissoziation10() : Emin(10.8)
{
    initialize_sum();
}

KSIntCalculatorHydrogenDissoziation10::KSIntCalculatorHydrogenDissoziation10(
    const KSIntCalculatorHydrogenDissoziation10& aCopy) :
    KSComponent(aCopy),
    Emin(aCopy.Emin)
{
    initialize_sum();
}

KSIntCalculatorHydrogenDissoziation10* KSIntCalculatorHydrogenDissoziation10::Clone() const
{
    return new KSIntCalculatorHydrogenDissoziation10(*this);
}

KSIntCalculatorHydrogenDissoziation10::~KSIntCalculatorHydrogenDissoziation10() = default;

// Parameters for a logarithmic expansion of the Crosssection which were originally obtained from
// a fit to experimental data.
// Taken from R.K Janev, W.D.Langer, K.Evans Jr., D.E.Post Jr. "Elementary Processes in Hydrogen-Helium-Plasmas"
// Springer (1987)
// ISBN 3-540-17588-1 Springer-Verlag Berlin Heidelberg New York
// ISBN 0-387-17588-1 Springer-Verlag New York Berlin Heidelberg
// page 44. Process 2.2.5
const double KSIntCalculatorHydrogenDissoziation10 ::CrossSectionParameters[9] = {-2.297914361e5,
                                                                                  5.303988579e5,
                                                                                  -5.316636672e5,
                                                                                  3.022690779e5,
                                                                                  -1.066224144e5,
                                                                                  2.389841369e4,
                                                                                  -3.324526406e3,
                                                                                  2.624761592e2,
                                                                                  -9.006246604};

void KSIntCalculatorHydrogenDissoziation10::CalculateCrossSection(const double anEnergy, double& aCrossSection)
{
    double lnsigma, lnE, lnEn;

    lnE = log(anEnergy);
    lnEn = 1.;
    lnsigma = 0.;

    if (anEnergy < Emin)
        aCrossSection = 0.;
    else {
        for (double CrossSectionParameter : CrossSectionParameters) {
            lnsigma += CrossSectionParameter * lnEn;
            lnEn = lnEn * lnE;
        }
        aCrossSection = 1.e-4 * exp(lnsigma);
    }

    return;
}

void KSIntCalculatorHydrogenDissoziation10::CalculateEloss(const double /*anEnergy*/, const double /*aTheta*/,
                                                           double& anEloss)
{
    anEloss = 10.;
}

/////////////////////////////////
/////   Dissoziation 15 eV	/////
/////////////////////////////////

KSIntCalculatorHydrogenDissoziation15::KSIntCalculatorHydrogenDissoziation15() : Emin(16.5)
{
    initialize_sum();
}

KSIntCalculatorHydrogenDissoziation15::KSIntCalculatorHydrogenDissoziation15(
    const KSIntCalculatorHydrogenDissoziation15& aCopy) :
    KSComponent(aCopy),
    Emin(aCopy.Emin)
{
    initialize_sum();
}

KSIntCalculatorHydrogenDissoziation15* KSIntCalculatorHydrogenDissoziation15::Clone() const
{
    return new KSIntCalculatorHydrogenDissoziation15(*this);
}

KSIntCalculatorHydrogenDissoziation15::~KSIntCalculatorHydrogenDissoziation15() = default;

// Parameters for a logarithmic expansion of the Crosssection which were originally obtained from
// a fit to experimental data.
// Taken from R.K Janev, W.D.Langer, K.Evans Jr., D.E.Post Jr. "Elementary Processes in Hydrogen-Helium-Plasmas"
// Springer (1987)
// ISBN 3-540-17588-1 Springer-Verlag Berlin Heidelberg New York
// ISBN 0-387-17588-1 Springer-Verlag New York Berlin Heidelberg
// page 46. Process 2.2.6
const double KSIntCalculatorHydrogenDissoziation15 ::CrossSectionParameters[9] = {-1.157041752e3,
                                                                                  1.501936271e3,
                                                                                  -8.6119387e2,
                                                                                  2.754926257e2,
                                                                                  -5.380465012e1,
                                                                                  6.573972423,
                                                                                  -4.912318139e-1,
                                                                                  2.054926773e-2,
                                                                                  -3.689035889e-4};


void KSIntCalculatorHydrogenDissoziation15::CalculateCrossSection(const double anEnergy, double& aCrossSection)
{
    double lnsigma, lnE, lnEn;

    lnE = log(anEnergy);
    lnEn = 1.;
    lnsigma = 0.;

    if (anEnergy < Emin)
        aCrossSection = 0.;
    else {
        for (double CrossSectionParameter : CrossSectionParameters) {
            lnsigma += CrossSectionParameter * lnEn;
            lnEn = lnEn * lnE;
        }
        aCrossSection = 1.e-4 * exp(lnsigma);
    }

    return;
}

void KSIntCalculatorHydrogenDissoziation15::CalculateEloss(const double /*anEnergy*/, const double /*aTheta*/,
                                                           double& anEloss)
{
    anEloss = 15.3;
}

/////////////////////////////////
////  Electronic Excitation  ////
/////////////////////////////////

KSIntCalculatorHydrogenExcitationElectronic::KSIntCalculatorHydrogenExcitationElectronic() :
    Emin(13.06),
    FrankCondonSum(35.86 + 40.05 + 6.58 + 2.26 + 9.61 + 4.08 + 1.54),
    ExcitationSum(6.58 + 2.26 + 9.61 + 4.08 + 1.54),
    ExcitationProbability(ExcitationSum / FrankCondonSum),
    pmax(9.61),
    nElectronicStates(5)
{}

KSIntCalculatorHydrogenExcitationElectronic::KSIntCalculatorHydrogenExcitationElectronic(
    const KSIntCalculatorHydrogenExcitationElectronic& aCopy) :
    KSComponent(aCopy),
    Emin(aCopy.Emin),
    FrankCondonSum(aCopy.FrankCondonSum),
    ExcitationSum(aCopy.ExcitationSum),
    pmax(aCopy.pmax),
    nElectronicStates(aCopy.nElectronicStates)
{}

KSIntCalculatorHydrogenExcitationElectronic* KSIntCalculatorHydrogenExcitationElectronic::Clone() const
{
    return new KSIntCalculatorHydrogenExcitationElectronic(*this);
}

KSIntCalculatorHydrogenExcitationElectronic::~KSIntCalculatorHydrogenExcitationElectronic() = default;

// Energy values of the excited electronic states:
//  (from Mol. Phys. 41 (1980) 1501, in eV)
const double KSIntCalculatorHydrogenExcitationElectronic ::EnergyLevels[7] =
    {12.73, 13.2, 14.77, 15.3, 14.93, 15.4, 13.06};

// Probability numbers of the electronic states:
//  (from testelectron7.c calculation )
const double KSIntCalculatorHydrogenExcitationElectronic::FrankCondonFactors[7] =
    {35.86, 40.05, 6.58, 2.26, 9.61, 4.08, 1.54};

void KSIntCalculatorHydrogenExcitationElectronic::CalculateCrossSection(const double anEnergy, double& aCrossSection)
{
    if (anEnergy < Emin)
        aCrossSection = 0.;
    else {
        aCrossSection = 4. * katrin::KConst::Pi() * katrin::KConst::BohrRadiusSquared() * katrin::KConst::ERyd_eV() /
                        anEnergy * (0.80 * log(anEnergy / katrin::KConst::ERyd_eV()) + 0.28);
        aCrossSection *= ExcitationProbability;
    }

    return;
}

void KSIntCalculatorHydrogenExcitationElectronic::CalculateEloss(const double /*anEnergy*/, const double /*aTheta*/,
                                                                 double& anEloss)
{
    double tRandom = KRandom::GetInstance().Uniform(0., ExcitationSum, false, true);

    int i;
    for (i = 2; i < 7; i++) {
        tRandom -= FrankCondonFactors[i];
        if (tRandom < 0)
            break;
    }

    anEloss = EnergyLevels[i];
}

/////////////////////////////////
/////	  New Ionisation	/////
/////////////////////////////////

KSIntCalculatorHydrogenIonisation::KSIntCalculatorHydrogenIonisation() :
    CrossParam_A1(0.74),  //
    CrossParam_A2(0.87),  //
    CrossParam_A3(-0.6),  //  Dimensonless Parameters of the Model
    CrossExponent(2.4),   //  in REF
    beta(0.6),            //
    gamma(10.),           //
    G_5(0.33),            //
    g_b(2.9),

    BindingEnergy(15.43),  // Hydrogen Binding Energy in eV

    // Normalisation Factor of the total ionisation crosssection
    // S = 4\pi a_0^2 N\frac{R}{I}^2
    Normalisation(4 * katrin::KConst::Pi() * 2 * katrin::KConst::BohrRadiusSquared() * katrin::KConst::ERyd_eV() *
                  katrin::KConst::ERyd_eV() / (BindingEnergy * BindingEnergy))
{}

KSIntCalculatorHydrogenIonisation::KSIntCalculatorHydrogenIonisation(
    const double aCrossParam_A1, const double aCrossParam_A2, const double aCrossParam_A3, const double aCrossExponent,
    const double abeta, const double agamma, const double aG_5, const double ag_b, const double aBindingEnergy)
{
    CrossParam_A1 = aCrossParam_A1;
    CrossParam_A2 = aCrossParam_A2;
    CrossParam_A3 = aCrossParam_A3;
    CrossExponent = aCrossExponent;
    beta = abeta;
    gamma = agamma;
    G_5 = aG_5;
    g_b = ag_b;
    BindingEnergy = aBindingEnergy;
    Normalisation = 4 * katrin::KConst::Pi() * 2 * katrin::KConst::BohrRadiusSquared() * katrin::KConst::ERyd_eV() *
                    katrin::KConst::ERyd_eV() / (BindingEnergy * BindingEnergy);
}

KSIntCalculatorHydrogenIonisation::KSIntCalculatorHydrogenIonisation(const KSIntCalculatorHydrogenIonisation& aCopy) :
    KSComponent(aCopy)
{
    CrossParam_A1 = aCopy.CrossParam_A1;
    CrossParam_A2 = aCopy.CrossParam_A2;
    CrossParam_A3 = aCopy.CrossParam_A3;
    CrossExponent = aCopy.CrossExponent;
    beta = aCopy.beta;
    gamma = aCopy.gamma;
    G_5 = aCopy.G_5;
    g_b = aCopy.g_b;
    BindingEnergy = aCopy.BindingEnergy;
    Normalisation = aCopy.Normalisation;
}

KSIntCalculatorHydrogenIonisation* KSIntCalculatorHydrogenIonisation::Clone() const
{
    return new KSIntCalculatorHydrogenIonisation(*this);
}

KSIntCalculatorHydrogenIonisation::~KSIntCalculatorHydrogenIonisation() = default;

void KSIntCalculatorHydrogenIonisation::CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection)
{
    CalculateCrossSection(aParticle.GetKineticEnergy_eV(), aCrossSection);
    return;
}


void KSIntCalculatorHydrogenIonisation::ExecuteInteraction(const KSParticle& anInitialParticle,
                                                           KSParticle& aFinalParticle, KSParticleQueue& aSecondaries)
{
    double tInitialKineticEnergy = anInitialParticle.GetKineticEnergy_eV();
    KThreeVector tInitialDirection = anInitialParticle.GetMomentum();

    // outgoing primary

    double tReducedInitialEnergy = tInitialKineticEnergy / BindingEnergy;
    double tReducedFinalEnergy;
    double tTheta;
    double tPhi;

    CalculateFinalEnergy(tReducedInitialEnergy, tReducedFinalEnergy);
    CalculateTheta(tReducedInitialEnergy, tReducedFinalEnergy, tTheta);

    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    KThreeVector tOrthogonalOne = tInitialDirection.Orthogonal();
    KThreeVector tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    KThreeVector tFinalDirection =
        tInitialDirection.Magnitude() *
        (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
         cos(tTheta) * tInitialDirection.Unit());

    aFinalParticle = anInitialParticle;
    aFinalParticle.SetMomentum(tFinalDirection);

    if (tReducedFinalEnergy < (tReducedInitialEnergy - 1.) / 2.) {
        tReducedFinalEnergy = tReducedInitialEnergy - tReducedFinalEnergy - 1.;
    }

    aFinalParticle.SetKineticEnergy_eV(tReducedFinalEnergy * BindingEnergy);

    fStepNInteractions = 1;
    fStepEnergyLoss = (tInitialKineticEnergy - tReducedFinalEnergy * BindingEnergy);

    // outgoing secondary

    tTheta = acos(KRandom::GetInstance().Uniform(-1., 1.));
    tPhi = KRandom::GetInstance().Uniform(0., 2. * katrin::KConst::Pi());

    tOrthogonalOne = tInitialDirection.Orthogonal();
    tOrthogonalTwo = tInitialDirection.Cross(tOrthogonalOne);
    tFinalDirection = tInitialDirection.Magnitude() *
                      (sin(tTheta) * (cos(tPhi) * tOrthogonalOne.Unit() + sin(tPhi) * tOrthogonalTwo.Unit()) +
                       cos(tTheta) * tInitialDirection.Unit());

    KSParticle* tSecondary = KSParticleFactory::GetInstance().Create(11);
    (*tSecondary) = anInitialParticle;
    tSecondary->SetMomentum(tFinalDirection);
    tSecondary->SetKineticEnergy_eV((tReducedInitialEnergy - tReducedFinalEnergy - 1.) * BindingEnergy);
    tSecondary->SetLabel(GetName());

    aSecondaries.push_back(tSecondary);

    return;
}

void KSIntCalculatorHydrogenIonisation::CalculateFinalEnergy(const double aReducedInitalEnergy,
                                                             double& aReducedFinalEnergy)
{
    double tReducedFinalEnergy;
    double tCrossSection = 0;


    double sigma_max;
    CalculateEnergyDifferentialCrossSection(aReducedInitalEnergy, aReducedInitalEnergy - 1.01, sigma_max);

    double sigma_min;
    CalculateEnergyDifferentialCrossSection(aReducedInitalEnergy, (aReducedInitalEnergy - 1) / 2, sigma_min);

    while (true) {
        tReducedFinalEnergy =
            KRandom::GetInstance().Uniform((aReducedInitalEnergy - 1.) / 2., aReducedInitalEnergy - 1., false, true);

        CalculateEnergyDifferentialCrossSection(aReducedInitalEnergy, tReducedFinalEnergy, tCrossSection);

        double tRandom = KRandom::GetInstance().Uniform(0.,
                                                        2. * (sigma_max - sigma_min) / (aReducedInitalEnergy - 1.) *
                                                            (tReducedFinalEnergy - (aReducedInitalEnergy - 1.) / 2.),
                                                        false,
                                                        true);

        if (tRandom < tCrossSection)
            break;
    }

    aReducedFinalEnergy = tReducedFinalEnergy;
}

void KSIntCalculatorHydrogenIonisation::CalculateTheta(const double aReducedInitialEnergy,
                                                       const double aReducedFinalEnergy, double& aTheta)
{
    double tTheta;
    double tCrossSection = 0;

    double sigma_max;
    CalculateDoublyDifferentialCrossSection(aReducedInitialEnergy,
                                            aReducedFinalEnergy,
                                            1.,  // == Cos(0.)
                                            sigma_max);
    sigma_max *= 1.5;

    while (true) {
        tTheta = KRandom::GetInstance().Uniform(0., katrin::KConst::Pi(), false, true);

        CalculateDoublyDifferentialCrossSection(aReducedInitialEnergy, aReducedFinalEnergy, cos(tTheta), tCrossSection);

        if (KRandom::GetInstance().Uniform(0., sigma_max, false, true) < tCrossSection)
            break;
    }

    aTheta = tTheta;
}

void KSIntCalculatorHydrogenIonisation::CalculateCrossSection(const double anEnergy, double& aCrossSection)
{
    // the reduced Energy is the Energy in units of the Binding Energy
    double tReducedEnergy = anEnergy / BindingEnergy;

    if (anEnergy < BindingEnergy)
        aCrossSection = 0;
    else {
        // \sigma_{\mathrm{ion}}(t) = S\cdot F(t)\cdot g_1(t)
        aCrossSection = Normalisation * Formfactor(tReducedEnergy) * g_1(tReducedEnergy);
    }
}

void KSIntCalculatorHydrogenIonisation::CalculateEnergyDifferentialCrossSection(const double aReducedInitialEnergy,
                                                                                const double aReducedFinalEnergy,
                                                                                double& aCrossSection)
{
    // \sigma(t, w) = G_1(t,w)\left[g_BE(t,w) + G_4(t,w) g_b\right]

    aCrossSection =
        G_1(aReducedInitialEnergy, aReducedFinalEnergy) *
        (g_BE(aReducedInitialEnergy, aReducedFinalEnergy) + G_4(aReducedInitialEnergy, aReducedFinalEnergy) * g_b);
}

void KSIntCalculatorHydrogenIonisation::CalculateDoublyDifferentialCrossSection(const double aReducedInitialEnergy,
                                                                                const double aReducedFinalEnergy,
                                                                                const double aCosTheta,
                                                                                double& aCrossSection)
{
    // \sigma(t, w, \theta) = G_1(t, w)\left[f_{BE}(t, w, \theta) + G_4(t, w)f_b(t, w, \theta)\right]

    aCrossSection = G_1(aReducedInitialEnergy, aReducedFinalEnergy) *
                    (f_BE(aReducedInitialEnergy, aReducedFinalEnergy, aCosTheta) +
                     G_4(aReducedInitialEnergy, aReducedFinalEnergy) * f_b(aCosTheta));
}

double KSIntCalculatorHydrogenIonisation::Formfactor(const double aReducedEnergy)
{
    // F(t) = ( A_1 * ln t + A_2 + \frac{A_3}{t} ) / t
    return (CrossParam_A1 * log(aReducedEnergy) + CrossParam_A2 + CrossParam_A3 / aReducedEnergy) / aReducedEnergy;
}

double KSIntCalculatorHydrogenIonisation::g_1(const double aReducedEnergy)
{
    // g_1(t) = \frac{1-t^{1-n}}{n-1}-\left[\frac{2}{t+1}^{\frac{n}{2}}\cdot\frac{1-t^{1-\frac{n}{2}}}{n-2}
    return (1. - pow(aReducedEnergy, 1 - CrossExponent)) / (CrossExponent - 1.) -
           pow(2. / (aReducedEnergy + 1), CrossExponent / 2.) * (1 - pow(aReducedEnergy, 1. - CrossExponent / 2.)) /
               (CrossExponent - 2.);
}

double KSIntCalculatorHydrogenIonisation::g_BE(const double aReducedInitialEnergy, const double aReducedFinalEnergy)
{
    // g_{BE}(w,t) = 2\piG_3\left[\tan^{-1}\left(\frac{1-G_2}{G_3}\right)+\tan^{-1}\left(\frac{1+G_2}{G_3}\right)\right]

    return 2. * katrin::KConst::Pi() * G_3(aReducedInitialEnergy, aReducedFinalEnergy) *
           (atan((1. - G_2(aReducedInitialEnergy, aReducedFinalEnergy)) /
                 G_3(aReducedInitialEnergy, aReducedFinalEnergy)) +
            atan((1. + G_2(aReducedInitialEnergy, aReducedFinalEnergy)) /
                 G_3(aReducedInitialEnergy, aReducedFinalEnergy)));
}

double KSIntCalculatorHydrogenIonisation::f_1(const double aReducedInitialEnergy, const double aReducedFinalEnergy)
{
    // f_1( w, t ) = \frac{1}{(w+1)^n} + \frac{1}{(t-w)^n} - \frac{1}{[(w+1)(t-w)]^{\frac{n}{2}}}

    return 1. / (pow(aReducedFinalEnergy + 1, CrossExponent)) +
           1. / (pow(aReducedInitialEnergy - aReducedFinalEnergy, CrossExponent)) -
           1. / (pow((aReducedFinalEnergy + 1) * (aReducedInitialEnergy - aReducedFinalEnergy), CrossExponent / 2.));
}

double KSIntCalculatorHydrogenIonisation::f_b(const double aCosTheta)
{
    // f_b = \frac{1}{1 + [(\cos\theta+1)/G_5]^2 }

    return 1. / (1. + pow((aCosTheta + 1.) / G_5, 2));
}

double KSIntCalculatorHydrogenIonisation::f_BE(const double aReducedInitialEnergy, const double aReducedFinalEnergy,
                                               const double aCosTheta)
{
    // f_BE = \frac{1}{ 1 + [ ( \cos\theta - G_2 ) / G_3 ]^2}

    return 1. / (1. + pow((aCosTheta - G_2(aReducedInitialEnergy, aReducedFinalEnergy)) /
                              G_3(aReducedInitialEnergy, aReducedFinalEnergy),
                          2));
}

double KSIntCalculatorHydrogenIonisation::G_1(const double aReducedInitialEnergy, const double aReducedFinalEnergy)
{
    // G_1(t, w) = \frac{ S F(t) f_1(t, w)}{I( g_{BE}(t, w) + G_4(t, w)g_b)}

    return Normalisation * Formfactor(aReducedInitialEnergy) * f_1(aReducedInitialEnergy, aReducedFinalEnergy) /
           BindingEnergy /
           (g_BE(aReducedInitialEnergy, aReducedFinalEnergy) + G_4(aReducedInitialEnergy, aReducedFinalEnergy) * g_b);
}

double KSIntCalculatorHydrogenIonisation::G_2(const double aReducedInitialEnergy, const double aReducedFinalEnergy)
{
    // G_2(t, w) = \sqrt{\frac{w+1}{t}}
    return sqrt((aReducedFinalEnergy + 1.) / aReducedInitialEnergy);
}

double KSIntCalculatorHydrogenIonisation::G_3(const double aReducedInitialEnergy, const double aReducedFinalEnergy)
{
    // G_3(t, w) = \beta\sqrt{\frac{1-G_2^2}{w}}

    return beta * sqrt((1. - pow(G_2(aReducedInitialEnergy, aReducedFinalEnergy), 2)) / aReducedFinalEnergy);
}

double KSIntCalculatorHydrogenIonisation::G_4(const double aReducedInitialEnergy, const double aReducedFinalEnergy)
{
    // G_4(t, w) = \gamma\frac{(1-\frac{w}{t})^3}{t(w+1}

    return gamma * pow(1. - aReducedFinalEnergy / aReducedInitialEnergy, 3) /
           (aReducedInitialEnergy * (aReducedFinalEnergy + 1));
}

} /* namespace Kassiopeia */

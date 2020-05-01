#include "ElasticFerencCalculator.h"

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KRandom.h"
using katrin::KRandom;

#include "KConst.h"

namespace
{
void RandomArray(size_t aN, double* aArray)
{
    for (size_t i = 0; i < aN; ++i) {
        *(aArray + i) = KRandom::GetInstance().Uniform(0.0, 1.0, false, true);
    }
}
}  // namespace

namespace Kassiopeia
{
ElasticFerencCalculator::ElasticFerencCalculator() {}

ElasticFerencCalculator::~ElasticFerencCalculator()
{
    //not needed
}

double ElasticFerencCalculator::sigmaeltot(double anE)
{

    //probabilities for rotation: P0(0->2), P1(1->3), P2(2->0)
    double P0 = 0.144;
    double P1 = 0.73;
    double P2 = 0.126;
    //cross sections
    double sigmel, sigmvib, sigmrot02, sigmrot13, sigmrot20, sigmatot;

    if (anE > 0.045) {
        sigmel = sigmael(anE);
        sigmvib = sigmavib(anE);
        sigmrot02 = sigmarot02(anE);
        sigmrot13 = sigmarot13(anE);
        sigmrot20 = sigmarot20(anE);
        sigmatot = sigmel + sigmvib + P0 * sigmrot02 + P1 * sigmrot13 + P2 * sigmrot20;
    }
    else {
        sigmatot = sigmael(anE) + P2 * sigmarot20(anE);
    }

    return sigmatot;
}

double ElasticFerencCalculator::sigmael(double anE)
{

    const double e[14] = {0., 1.5, 5., 7., 10., 15., 20., 30., 60., 100., 150., 200., 300., 400.};
    const double s[14] = {9.6, 13., 15., 12., 10., 7., 5.6, 3.3, 1.1, 0.9, 0.5, 0.36, 0.23, 0.15};

    const double emass = 1. / (katrin::KConst::Alpha() * katrin::KConst::Alpha());
    const double a02 = katrin::KConst::BohrRadiusSquared();

    double gam, T;
    double sigma = 0.0;
    T = anE / 27.2;
    if (anE >= 400.) {
        gam = (emass + T) / emass;
        sigma = gam * gam * katrin::KConst::Pi() / (2. * T) * (4.2106 - 1. / T) * a02;
    }
    else {
        for (unsigned int i = 0; i <= 12; i++) {
            if (anE >= e[i] && anE < e[i + 1])
                sigma = 1.e-20 * (s[i] + (s[i + 1] - s[i]) * (anE - e[i]) / (e[i + 1] - e[i]));
        }
    }
    return sigma;
}

double ElasticFerencCalculator::sigmavib(double anE)
{

    double sigma = 0.0;
    unsigned int i;

    static double sigma1[8] = {0.0, 0.006, 0.016, 0.027, 0.033, 0.045, 0.057, 0.065};

    static double sigma2[9] = {0.065, 0.16, 0.30, 0.36, 0.44, 0.47, 0.44, 0.39, 0.34};

    static double sigma3[7] = {0.34, 0.27, 0.21, 0.15, 0.12, 0.08, 0.07};

    if (anE <= 0.5 || anE > 10.) {
        sigma = 0.;
    }
    else {
        if (anE >= 0.5 && anE < 1.0) {
            i = (anE - 0.5) / 0.1;
            sigma = 1.e-20 * (sigma1[i] + (sigma1[i + 1] - sigma1[i]) * (anE - 0.5 - i * 0.1) * 10.);
        }
        else {
            if (anE >= 1.0 && anE < 5.0) {
                i = (anE - 1.0) / 0.5;
                sigma = 1.e-20 * (sigma2[i] + (sigma2[i + 1] - sigma2[i]) * (anE - 1.0 - i * 0.5) * 2.);
            }
            else {
                i = (anE - 5.0) / 1.0;
                sigma = 1.e-20 * (sigma3[i] + (sigma3[i + 1] - sigma3[i]) * (anE - 5.0 - i * 1.0));
            }
        }
    }

    return sigma;
}

double ElasticFerencCalculator::sigmarot02(double anE)
{

    double sigma = 0.0;
    unsigned int i;

    // sigma1: 0.015 -- 0.03; 0.003
    // sigma2: 0.03 -- 0.1;   0.01
    // sigma1: 0.1 -- 1.0;    0.1
    // sigma2: 1.0 -- 5.0;    0.5
    // sigma3: 5.0 -- 10.0;   1.0
    // static double sigma1[6]={0.0,0.03,0.045,0.052,0.059,0.065};

    static double sigma2[8] = {0.065, 0.069, 0.073, 0.077, 0.081, 0.085, 0.088, 0.090};

    static double sigma3[10] = {0.09, 0.11, 0.15, 0.20, 0.26, 0.32, 0.39, 0.47, 0.55, 0.64};

    static double sigma4[9] = {0.64, 1.04, 1.37, 1.58, 1.70, 1.75, 1.76, 1.73, 1.69};

    static double sigma5[7] = {1.69, 1.58, 1.46, 1.35, 1.25, 1.16, 1.0};

    static double DeltaE = 0.045;

    if (anE <= DeltaE + 1.e-8 || anE > 10.) {
        sigma = 0.;
    }
    else {
        if (anE >= 0.045 && anE < 0.1) {
            i = (anE - 0.045) / 0.01;
            sigma = 1.e-20 * (sigma2[i] + (sigma2[i + 1] - sigma2[i]) * (anE - 0.045 - i * 0.01) * 100.);
        }
        else {
            if (anE >= 0.1 && anE < 1.0) {
                i = (anE - 0.1) / 0.1;
                sigma = 1.e-20 * (sigma3[i] + (sigma3[i + 1] - sigma3[i]) * (anE - 0.1 - i * 0.1) * 10.);
            }
            else {
                if (anE >= 1.0 && anE < 5.0) {
                    i = (anE - 1.0) / 0.5;
                    sigma = 1.e-20 * (sigma4[i] + (sigma4[i + 1] - sigma4[i]) * (anE - 1.0 - i * 0.5) * 2.);
                }
                else {
                    i = (anE - 5.0) / 1.0;
                    sigma = 1.e-20 * (sigma5[i] + (sigma5[i + 1] - sigma5[i]) * (anE - 5.0 - i * 1.0));
                }
            }
        }
    }

    return sigma;
}

double ElasticFerencCalculator::sigmarot13(double anE)
{

    double sigma = 0.0;
    unsigned int i;

    // sigma1: 0.025 -- 0.05;   0.005
    // sigma2: 0.05 -- 0.randomProbability1;     0.01
    // sigma3: 0.1 -- 1.0;      0.1
    // sigma4: 1.0 -- 5.0;      0.5
    // sigma5: 5.0 -- 10.0;     1.0
    //static double sigma1[6]={0.0,0.02,0.025,0.029,0.032,0.035};

    static double sigma2[6] = {0.035, 0.038, 0.041, 0.044, 0.047, 0.05};

    static double sigma3[10] = {0.05, 0.065, 0.09, 0.11, 0.14, 0.18, 0.21, 0.25, 0.29, 0.33};

    static double sigma4[9] = {0.33, 0.55, 0.79, 0.94, 1.01, 1.05, 1.05, 1.04, 1.01};

    static double sigma5[7] = {1.01, 0.95, 0.88, 0.81, 0.75, 0.69, 0.62};

    static double DeltaE = 0.075;

    if (anE <= DeltaE + 1.e-8 || anE > 10.) {
        sigma = 0.;
    }
    else {
        if (anE >= 0.075 && anE < 0.1) {
            i = (anE - 0.075) / 0.01;
            sigma = 1.e-20 * (sigma2[i] + (sigma2[i + 1] - sigma2[i]) * (anE - 0.075 - i * 0.01) * 100.);
        }
        else {
            if (anE >= 0.1 && anE < 1.0) {
                i = (anE - 0.1) / 0.1;
                sigma = 1.e-20 * (sigma3[i] + (sigma3[i + 1] - sigma3[i]) * (anE - 0.1 - i * 0.1) * 10.);
            }
            else {
                if (anE >= 1.0 && anE < 5.0) {
                    i = (anE - 1.0) / 0.5;
                    sigma = 1.e-20 * (sigma4[i] + (sigma4[i + 1] - sigma4[i]) * (anE - 1.0 - i * 0.5) * 2.);
                }
                else {
                    i = (anE - 5.0) / 1.0;
                    sigma = 1.e-20 * (sigma5[i] + (sigma5[i + 1] - sigma5[i]) * (anE - 5.0 - i * 1.0));
                }
            }
        }
    }
    return sigma;
}

double ElasticFerencCalculator::sigmarot20(double anE)
{

    double sigma = 0.0;
    double Ep = anE + 0.045;

    sigma = 1. / 5. * Ep / anE * sigmarot02(Ep);

    return sigma;
}

void ElasticFerencCalculator::randomel(double anE, double& anEloss, double& aTheta)
{
    double emass = 1. / (katrin::KConst::Alpha() * katrin::KConst::Alpha());
    double clight = 1. / katrin::KConst::Alpha();

    double H2molmass = 69.e6;

    double T, c, b, G, a, u[3], gam, K2, Gmax;

    double sigmel, sigmvib, sigmrot02, sigmrot13, sigmrot20, sigmatot;

    double P0 = 0.144;
    double P1 = 0.73;
    double P2 = 0.126;
    double Pel, Pvib, Prot02, Prot13, Prot20;
    double randomProbability = KRandom::GetInstance().Uniform();

    if (anE > 0.045) {
        sigmel = sigmael(anE);
        sigmvib = sigmavib(anE);
        sigmrot02 = sigmarot02(anE);
        sigmrot13 = sigmarot13(anE);
        sigmrot20 = sigmarot20(anE);
        sigmatot = sigmel + sigmvib + P0 * sigmrot02 + P1 * sigmrot13 + P2 * sigmrot20;
        Pel = sigmel / sigmatot;
        Pvib = sigmvib / sigmatot;
        Prot02 = P0 * sigmrot02 / sigmatot;
        Prot13 = P1 * sigmrot13 / sigmatot;
        Prot20 = P2 * sigmrot20 / sigmatot;
    }
    else {
        sigmel = sigmael(anE);
        sigmrot20 = sigmarot20(anE);
        sigmatot = sigmel + P2 * sigmrot20;
        Pel = sigmel / sigmatot;
        Prot20 = P2 * sigmrot20 / sigmatot;
        Pvib = Prot02 = Prot13 = 0.;
    }

    if (anE >= 250.)
        Gmax = 1.e-19;
    else if (anE < 250. && anE >= 150.)
        Gmax = 2.5e-19;
    else
        Gmax = 1.e-18;
    T = anE / 27.2;
    gam = 1. + T / (clight * clight);
    b = 2. / (1. + gam) / T;
    for (int i = 1; i < 5000; i++) {
        RandomArray(3, u);

        c = 1. + b - b * (2. + b) / (b + 2. * u[1]);
        K2 = 2. * T * (1. + gam) * fabs(1. - c);
        a = (4. + K2) * (4. + K2) / (gam * gam);
        G = a * DiffXSecEl(anE, c);
        if (G > Gmax * u[2])
            break;
    }
    aTheta = acos(c);

    if (randomProbability < Pel) {
        if (anE < 1.) {
            //check if electron won energy by elastic scattering on a molecule; this keeps electron energies around the gas temperature
            double rndNr = KRandom::GetInstance().Uniform();
            double rndAngle = KRandom::GetInstance().Uniform();

            //generation of molecule velocity by maxwell-boltzmann distribution
            double Gx = sqrt(-2. * log(rndNr)) * cos(2. * katrin::KConst::Pi() * rndAngle);
            double Gy = sqrt(-2. * log(rndNr)) * sin(2. * katrin::KConst::Pi() * rndAngle);
            double Gz = sqrt(-2. * log(KRandom::GetInstance().Uniform())) *
                        cos(2. * katrin::KConst::Pi() * KRandom::GetInstance().Uniform());

            //thermal velocity of gas molecules
            double T = 300.;  //gas temperature
            double sigmaT = sqrt(katrin::KConst::kB() * T / (2. * katrin::KConst::M_prot_kg()));
            KThreeVector MolVelocity(sigmaT * Gx, sigmaT * Gy, sigmaT * Gz);

            //new electron velocity vector and energy:

            //assume electron velocity along z
            KThreeVector ElVelocity(0., 0., sqrt(2. * anE * katrin::KConst::Q() / katrin::KConst::M_el_kg()));
            //relative velocity electron-molecule
            KThreeVector RelativeVelocity = ElVelocity - MolVelocity;
            //transformation into CMS
            KThreeVector CMSVelocity =
                (katrin::KConst::M_el_kg() / (katrin::KConst::M_el_kg() + katrin::KConst::M_prot_kg()) * ElVelocity +
                 2. * katrin::KConst::M_prot_kg() / (katrin::KConst::M_el_kg() + katrin::KConst::M_prot_kg()) *
                     MolVelocity);
            //generation of random direction
            KThreeVector Random(KRandom::GetInstance().Uniform(),
                                KRandom::GetInstance().Uniform(),
                                KRandom::GetInstance().Uniform());
            //new electron velocity
            ElVelocity = katrin::KConst::M_prot_kg() / (katrin::KConst::M_prot_kg() + katrin::KConst::M_el_kg()) *
                             RelativeVelocity.Magnitude() * Random +
                         CMSVelocity;
            anEloss = anE - katrin::KConst::M_el_kg() / (2. * katrin::KConst::Q()) * ElVelocity.Magnitude() *
                                ElVelocity.Magnitude();
            //check if electron won energy due to molecule scattering
            if (anEloss < 0.) {
                return;
            }
            else {
                //without energy gain due to molecule scattering
                anEloss = 2. * emass / H2molmass * (1. - c) * anE;
            }
        }
        else {
            //without energy gain due to molecule scattering
            anEloss = 2. * emass / H2molmass * (1. - c) * anE;
        }
    }
    else {
        if (randomProbability >= Pel && randomProbability < Pel + Pvib) {
            anEloss = 0.5;
            //std::cout <<"Eloss vib: "<<anEloss<<std::endl;
        }
        else {
            if (randomProbability >= Pel + Pvib && randomProbability < Pel + Pvib + Prot02) {
                anEloss = 0.045;
                //std::cout <<"anEloss rot02: "<<anEloss<<std::endl;
            }
            else {
                if (randomProbability >= Pel + Pvib + Prot02 && randomProbability < Pel + Pvib + Prot02 + Prot13) {
                    anEloss = 0.075;
                    //std::cout <<"anEloss rot13: "<<anEloss<<std::endl;
                }
                else {
                    anEloss = -0.045;
                    //std::cout <<"anEloss rot20: "<<anEloss<<std::endl;
                }
            }
        }
    }
    return;
}

double ElasticFerencCalculator::DiffXSecEl(double anE, double cosTheta)
{

    //double a02=28.e-22;   // Bohr radius squared
    double a02 = katrin::KConst::BohrRadiusSquared();

    double clight = 1. / katrin::KConst::Alpha();  // velocity of light in atomic units is 1/ alpha

    double Cel[50] = {-0.512, -0.512, -0.509, -0.505, -0.499, -0.491, -0.476, -0.473, -0.462, -0.452,
                      -0.438, -0.422, -0.406, -0.388, -0.370, -0.352, -0.333, -0.314, -0.296, -0.277,
                      -0.258, -0.239, -0.221, -0.202, -0.185, -0.167, -0.151, -0.135, -0.120, -0.105,
                      -0.092, -0.070, -0.053, -0.039, -0.030, -0.024, -0.019, -0.016, -0.014, -0.013,
                      -0.012, -0.009, -0.008, -0.006, -0.005, -0.004, -0.003, -0.002, -0.002, -0.001};

    double e[10] = {0., 3., 6., 12., 20., 32., 55., 85., 150., 250.};

    double t[10] = {0., 10., 20., 30., 40., 60., 80., 100., 140., 180.};

    double D[9][10] = {{2.9, 2.70, 2.5, 2.10, 1.80, 1.2000, 0.900, 1.0000, 1.600, 1.9},
                       {4.2, 3.60, 3.1, 2.50, 1.90, 1.1000, 0.800, 0.9000, 1.300, 1.4},
                       {6.0, 4.40, 3.2, 2.30, 1.80, 1.1000, 0.700, 0.5400, 0.500, 0.6},
                       {6.0, 4.10, 2.8, 1.90, 1.30, 0.6000, 0.300, 0.1700, 0.160, 0.23},
                       {4.9, 3.20, 2.0, 1.20, 0.80, 0.3000, 0.150, 0.0900, 0.050, 0.05},
                       {5.2, 2.50, 1.2, 0.64, 0.36, 0.1300, 0.050, 0.0300, 0.016, 0.02},
                       {4.0, 1.70, 0.7, 0.30, 0.16, 0.0500, 0.020, 0.0130, 0.010, 0.01},
                       {2.8, 1.10, 0.4, 0.15, 0.07, 0.0200, 0.010, 0.0070, 0.004, 0.003},
                       {1.2, 0.53, 0.2, 0.08, 0.03, 0.0074, 0.003, 0.0016, 0.001, 0.0008}};

    double T, K2, K, d, st1, st2, DH, gam, CelK, Ki, theta;
    double Delreturn = -1.0;
    int i, j;
    T = anE / 27.2;
    if (anE >= 250.) {
        gam = 1. + T / (clight * clight);  // relativistic correction factor
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

        Delreturn = 2. * gam * gam * DH * (1. + sin(K * d) / (K * d)) * (1. + CelK);
    }  //end if anE>=250
    else {
        theta = acos(cosTheta) * 180. / katrin::KConst::Pi();
        for (i = 0; i <= 8; i++)
            if (anE >= e[i] && anE < e[i + 1])
                for (j = 0; j <= 8; j++)
                    if (theta >= t[j] && theta < t[j + 1])
                        Delreturn = 1.e-20 * (D[i][j] + (D[i][j + 1] - D[i][j]) * (theta - t[j]) / (t[j + 1] - t[j]));
    }
    return Delreturn;

}  //end DiffXSecEl
}  // namespace Kassiopeia

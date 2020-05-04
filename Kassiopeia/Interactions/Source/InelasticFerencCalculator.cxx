#include "InelasticFerencCalculator.h"

#include "KTextFile.h"

#include <fstream>
using std::fstream;

#include "KConst.h"
#include "KRandom.h"
#include "KSInteractionsMessage.h"
#include "TMath.h"

using namespace std;
using namespace katrin;

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
InelasticFerencCalculator::InelasticFerencCalculator() :
    fDataFile(),
    fBindingEnergy(),
    fOrbitalEnergy(),
    fNOccupation(),
    fIonizationEnergy(0),
    fMoleculeType(""),
    fMinimum(-1)
{}

InelasticFerencCalculator::~InelasticFerencCalculator()
{
    //not needed
}

void InelasticFerencCalculator::setmolecule(const std::string& aMolecule)
{
    if (aMolecule.length() == 0) {
        return;
    }

    fMoleculeType = aMolecule;
    fDataFile = katrin::CreateDataTextFile(fMoleculeType + string(".txt"));
    if (ReadData() == false) {
        intmsg(eError) << "scattering data file corresponding to molecule <" << aMolecule << "> not found" << eom;
    }
    return;
}
double InelasticFerencCalculator::GetIonizationEnergy()
{
    return fIonizationEnergy;
}

double InelasticFerencCalculator::sigmaexc(double anE)
{
    //TODO: move constants somewhere. precision?

    const double a02 = katrin::KConst::BohrRadiusSquared();
    // double a02= 28.e-22; //used several times

    const double R = 13.6;  //hydrogen ionisation? precision?

    double sigma;
    if (anE <
        11.18)  //no excitation below this energy, otherwise your primary can have negative energy after the process
        sigma = 0.0;
    else if (anE >= 11.18 && anE <= 250.)
        sigma = sigmaBC(anE) + sigmadiss10(anE) + sigmadiss15(anE);
    else
        sigma = 4. * katrin::KConst::Pi() * a02 * R / anE * (0.80 * log(anE / R) + 0.28);
    //    sigma=sigmainel(anE)-sigmaion(anE);
    return sigma;
}

void InelasticFerencCalculator::randomexc(
    double anE, double& Eloss, double& Theta)  //Todo: move static stuff to constructor. constants and precision?
{

    static int iff = 0;
    static double sum[1001];
    static double fmax;

    double Ecen = 12.6 / 27.21;

    double T, c, u[3], K, xmin, ymin, ymax, x, y, fy, dy, pmax;
    double D, Dmax;
    int i, j, n, N, v;
    // Energy values of the excited electronic states:
    //  (from Mol. Phys. 41 (1980) 1501, in Hartree atomic units)
    double En[7] = {12.73 / 27.2, 13.2 / 27.2, 14.77 / 27.2, 15.3 / 27.2, 14.93 / 27.2, 15.4 / 27.2, 13.06 / 27.2};
    // Probability numbers of the electronic states:
    //  (from testelectron7.c calculation )
    double p[7] = {35.86, 40.05, 6.58, 2.26, 9.61, 4.08, 1.54};
    // Energy values of the B vibrational states:
    //   (from: Phys. Rev. A51 (1995) 3745 , in Hartree atomic units)
    double EB[28] = {0.411, 0.417, 0.423, 0.428, 0.434, 0.439, 0.444, 0.449, 0.454, 0.459, 0.464, 0.468, 0.473, 0.477,
                     0.481, 0.485, 0.489, 0.493, 0.496, 0.500, 0.503, 0.507, 0.510, 0.513, 0.516, 0.519, 0.521, 0.524};
    // Energy values of the C vibrational states:
    //   (from: Phys. Rev. A51 (1995) 3745 , in Hartree atomic units)
    double EC[14] = {0.452, 0.462, 0.472, 0.481, 0.490, 0.498, 0.506, 0.513, 0.519, 0.525, 0.530, 0.534, 0.537, 0.539};
    // Franck-Condon factors of the B vibrational states:
    //   (from: Phys. Rev. A51 (1995) 3745 )
    double pB[28] = {4.2e-3, 1.5e-2, 3.0e-2, 4.7e-2, 6.3e-2, 7.3e-2, 7.9e-2, 8.0e-2, 7.8e-2, 7.3e-2,
                     6.6e-2, 5.8e-2, 5.1e-2, 4.4e-2, 3.7e-2, 3.1e-2, 2.6e-2, 2.2e-2, 1.8e-2, 1.5e-2,
                     1.3e-2, 1.1e-2, 8.9e-3, 7.4e-3, 6.2e-3, 5.2e-3, 4.3e-3, 3.6e-3};
    // Franck-Condon factors of the C vibrational states:
    //   (from: Phys. Rev. A51 (1995) 3745 )
    double pC[14] = {1.2e-1,
                     1.9e-1,
                     1.9e-1,
                     1.5e-1,
                     1.1e-1,
                     7.5e-2,
                     5.0e-2,
                     3.3e-2,
                     2.2e-2,
                     1.4e-2,
                     9.3e-3,
                     6.0e-3,
                     3.7e-3,
                     1.8e-3};

    T = 20000. / 27.2;
    //
    xmin = Ecen * Ecen / (2. * T);
    ymin = log(xmin);
    ymax = log(8. * T + xmin);
    dy = (ymax - ymin) / 1000.;

    // Initialization of the sum[] vector, and fmax calculation://TODO move to constructor
    if (iff == 0) {
        fmax = 0;
        for (i = 0; i <= 1000; i++) {
            y = ymin + dy * i;
            K = exp(y / 2.);
            sum[i] = sumexc(K);
            if (sum[i] > fmax)
                fmax = sum[i];
        }
        fmax = 1.05 * fmax;
        iff = 1;
    }
    //
    //  Scattering angle Theta generation:
    //
    T = anE / 27.2;
    if (anE >= 100.) {
        xmin = Ecen * Ecen / (2. * T);
        ymin = log(xmin);
        ymax = log(8. * T + xmin);
        dy = (ymax - ymin) / 1000.;
        // Generation of y values with the Neumann acceptance-rejection method:
        for (j = 1; j < 5000; j++) {
            //subrn(u,2);
            RandomArray(3, u);
            y = ymin + (ymax - ymin) * u[1];
            K = exp(y / 2.);
            fy = sumexc(K);
            if (fmax * u[2] < fy)
                break;
        }
        // Calculation of c=cos(Theta) and Theta:
        x = exp(y);
        c = 1. - (x - xmin) / (4. * T);
        Theta = acos(c) * 180. / katrin::KConst::Pi();
    }
    else {
        if (anE <= 25.)
            Dmax = 60.;
        else if (anE > 25. && anE <= 35.)
            Dmax = 95.;
        else if (anE > 35. && anE <= 50.)
            Dmax = 150.;
        else
            Dmax = 400.;
        for (j = 1; j < 5000; j++) {
            //subrn(u,2);
            RandomArray(3, u);
            c = -1. + 2. * u[1];
            D = DiffXSecExc(anE, c) * 1.e22;
            if (Dmax * u[2] < D)
                break;
        }
        Theta = acos(c) * 180. / katrin::KConst::Pi();
    }
    // Energy loss Eloss generation:

    // First we generate the electronic state, using the Neumann
    // acceptance-rejection method for discrete distribution:
    N = 7;        // the number of electronic states in our calculation
    pmax = p[1];  // the maximum of the p[] values
    for (j = 1; j < 5000; j++) {
        RandomArray(3, u);
        //subrn(u,2);
        n = (int) (N * u[1]);  //WOLF: conversion int-> double -> int
        if (u[2] * pmax < p[n])
            break;
    }
    if (n < 0)
        n = 0;
    else if (n > 6)
        n = 6;

    if (n > 1)  // Bp, Bpp, D, Dp, EF states
    {
        Eloss = En[n] * 27.2;
        return;
    }
    if (n == 0)  // B state; we generate now a vibrational state,
    // using the Frank-Condon factors
    {
        N = 28;        // the number of B vibrational states in our calculation
        pmax = pB[7];  // maximum of the pB[] values
        for (j = 1; j < 5000; j++) {
            RandomArray(3, u);
            //subrn(u,2); //WOLF: conversion int->double
            v = (int) (N * u[1]);
            if (u[2] * pmax < pB[v])
                break;
        }
        if (v < 0)
            v = 0;
        if (v > 27)
            v = 27;
        Eloss = EB[v] * 27.2;
    }
    if (n == 1)  // C state; we generate now a vibrational state,
    // using the Franck-Condon factors
    {
        N = 14;        // the number of C vibrational states in our calculation
        pmax = pC[1];  // maximum of the pC[] values
        for (j = 1; j < 5000; j++) {
            RandomArray(3, u);
            //subrn(u,2);
            v = (int) (N * u[1]);  //WOLF> conversion int->double
            if (u[2] * pmax < pC[v])
                break;
        }
        if (v < 0)
            v = 0;
        else if (v > 13)
            v = 13;
        Eloss = EC[v] * 27.2;
    }
    return;

}  //end randomexc

//old version (HYDROGEN ONLY!)
/*
     double KScatterBasicInelasticCalculatorFerenc::sigmaion(double anE){ //TODO: move atomic units somewhere
     double B=15.43,U=15.98,R=13.6;
     //const double a02=0.28e-20;
     const double a02 = KaConst::BohrRadiusSquared();
     double sigma,t,u,S,r,lnt;
     if(anE<16.)
     sigma=1.e-40;
     else if(anE>=16. && anE<=250.)
     {
     t=anE/B;
     u=U/B;
     r=R/B;
     S=4.*KaConst::Pi()*a02*2.*r*r;
     lnt=log(t);
     sigma=S/(t+u+1.)*(lnt/2.*(1.-1./(t*t))+1.-1./t-lnt/(t+1.));
     }
     else
     sigma=4.*KaConst::Pi()*a02*R/anE*(0.82*log(anE/R)+1.3);
     return sigma;

     }
     */
double InelasticFerencCalculator::sigmaion(double anE)
{

    double sigma = 0.0;

    std::vector<double> CrossSections;
    double TotalCrossSection = 0.0;

    if (fBindingEnergy.size() == 0) {
        intmsg(eError) << "InelasticFerencCalculator::sigmaion" << ret;
        intmsg << "using unitialized calculator. quitting" << eom;
    }
    const double a02 = katrin::KConst::BohrRadiusSquared();
    const double ERyd = katrin::KConst::ERyd_eV();  //this does exist!

    //in old code:
    //const double ERyd =13.6;//Rydberg constant from EH2SCAT
    //or nancys value:
    //const double R=13.6057;//Rydberg constant

    if (anE > fOrbitalEnergy[fMinimum]) {
        //if (anE > 16.){//EH2scat sucks

        //for (std::vector<double>::iterator orbitalIt = fOrbitalEnergy.begin();
        //		orbitalIt != fOrbitalEnergy.end(); orbitalIt++){

        for (unsigned int io = 0; io < fOrbitalEnergy.size(); io++) {
            int i = 0;  //numbers the possible shells

            if (anE > fOrbitalEnergy[io]) {

                //???special case for hydrogen and anE>250 EV ????
                //is this an approximation, or is this acually more correct?
                if (fMoleculeType == "Hydrogen" && anE > 250.) {

                    sigma = 4. * katrin::KConst::Pi() * a02 * ERyd / anE * (0.82 * log(anE / ERyd) + 1.3);
                    CrossSections.push_back(4. * katrin::KConst::Pi() * a02 * ERyd / anE *
                                            (0.82 * log(anE / ERyd) + 1.3));

                    TotalCrossSection = CrossSections.at(i);
                }
                else {
                    double t = anE / fBindingEnergy.at(io);
                    double u = fOrbitalEnergy.at(io) / fBindingEnergy.at(io);
                    double r = ERyd / fBindingEnergy.at(io);
                    double S = 4. * katrin::KConst::Pi() * a02 * fNOccupation.at(io) * r * r;
                    double lnt = TMath::Log(t);

                    CrossSections.push_back(S / (t + u + 1.) *
                                            (lnt / 2. * (1. - 1. / (t * t)) + 1. - 1. / t - lnt / (t + 1.)));
                    TotalCrossSection += CrossSections.at(i);

                    sigma += S / (t + u + 1.) * (lnt / 2. * (1. - 1. / (t * t)) + 1. - 1. / t - lnt / (t + 1.));
                }
                i++;
            }
            else
                sigma = 1E-40;  //for eh2scat comparison!
        }

        //Determination of ionization energy
        //Random number
        double IonizationDice = KRandom::GetInstance().Uniform();
        //Decide from which shell the secondary electron is kicked out
        for (unsigned int i = 0; i < CrossSections.size(); i++) {
            IonizationDice -= CrossSections.at(i) / TotalCrossSection;
            if (IonizationDice < 0) {
                fIonizationEnergy = fBindingEnergy.at(i);

                intmsg_debug("InelasticFerencCalculator::sigmaion" << ret);
                intmsg_debug("ionization energy: " << CrossSections.at(i) << eom);

                break;
            }
        }
    }

    else
        sigma = 1E-40;  //why not 0?
    return sigma;
}

bool InelasticFerencCalculator::ReadData()
{

    if (fDataFile->Open(KFile::eRead) == true) {

        fstream& inputfile = *(fDataFile->File());
        double aTemp, anotherTemp;
        int aTempInt;

        fBindingEnergy.clear();
        fOrbitalEnergy.clear();
        fNOccupation.clear();

        while (!inputfile.eof()) {

            //does this remove comments? should work.
            Char_t c = inputfile.peek();
            if (c >= '0' && c < '9') {
                inputfile >> aTemp >> anotherTemp >> aTempInt;
                intmsg_debug("InelasticFerencCalculator::ReadData " << ret);
                intmsg_debug(aTemp << " " << anotherTemp << " " << aTempInt << eom);
                fBindingEnergy.push_back(aTemp);
                fOrbitalEnergy.push_back(anotherTemp);
                fNOccupation.push_back(aTempInt);
            }
            else {
                Char_t dump[200];
                inputfile.getline(dump, 200);
                intmsg_debug("InelasticFerencCalculator::ReadData " << ret);
                intmsg_debug("dumping " << dump << " because " << c << " is not a number" << eom);
                continue;
            }
        }
    }
    else {

        intmsg(eError) << "InelasticFerencCalculator::ReadData: FATAL ERROR reading scattering data: inputfile <"
                       << fMoleculeType << ".txt> not found or molecule type not supported. " << eom;
    }
    fDataFile->Close();

    return FindMinimum();
}

bool InelasticFerencCalculator::FindMinimum()
{
    double aMinimum = 999999.99;
    for (unsigned int io = 0; io < fOrbitalEnergy.size(); io++) {
        if (aMinimum > fOrbitalEnergy[io])
            fMinimum = (int) io;
    }
    if (fMinimum >= 0)
        return true;
    else
        return false;
}

void InelasticFerencCalculator::randomion(double anE, double& Eloss, double& Theta)
{
    // << "Eloss Computation" << endl;

    //double Ei=15.45/27.21;
    double IonizationEnergy_eV = GetIonizationEnergy();        //ionization energy in eV
    double IonizationEnergy_au = IonizationEnergy_eV / 27.21;  //ionization energy in atomic units
    double c, b, u[3], K, xmin, ymin, ymax, x, y, T, G, W, Gmax;
    double q, h, F, Fmin, Fmax, Gp, Elmin, Elmax, qmin, qmax, El, wmax;
    double WcE, Jstarq, WcstarE, w, D2ion;
    //int j;
    double K2, KK, fE, kej, ki, kf, Rex, arg, arctg;
    //int i;
    double st1, st2;
    double Theta_deg;
    //
    // I. Generation of Theta
    // -----------------------
    Gmax = 1.e-20;
    if (anE < 200.)
        Gmax = 2.e-20;
    T = anE / 27.21;
    xmin = IonizationEnergy_au * IonizationEnergy_au / (2. * T);
    b = xmin / (4. * T);
    ymin = log(xmin);
    ymax = log(8. * T + xmin);
    // Generation of y values with the Neumann acceptance-rejection method:
    for (int j = 1; j < 5000; j++) {
        //subrn(u,2);
        RandomArray(3, u);
        y = ymin + (ymax - ymin) * u[1];
        K = exp(y / 2.);
        c = 1. + b - K * K / (4. * T);
        G = K * K * (DiffXSecInel(anE, c) - DiffXSecExc(anE, c));
        if (Gmax * u[2] < G)
            break;
    }
    // y --> x --> c --> Theta
    x = exp(y);
    c = 1. - (x - xmin) / (4. * T);
    Theta = acos(c);
    Theta_deg = Theta * 180. / katrin::KConst::Pi();

    //
    // II. Generation of Eloss, for fixed Theta
    // ----------------------------------------
    //
    // For anE<=200 eV we use subr. gensecelen
    //   (in this case no correlation between Theta and Eloss)
    if (anE <= 200.) {
        gensecelen(anE, W);
        Eloss = IonizationEnergy_eV + W;
        return;
    }
    // For Theta>=20 the free electron model is used
    //   (with full correlation between Theta and Eloss)
    if (Theta_deg >= 20.) {
        Eloss = anE * (1. - c * c);
        if (Eloss < IonizationEnergy_eV + 0.05)
            Eloss = IonizationEnergy_eV + 0.05;
        return;
    }
    // For anE>200 eV and Theta<20: analytical first Born approximation
    //   formula of Bethe for H atom (with modification for H2)
    //
    // Calc. of wmax:
    if (Theta_deg >= 0.7)
        wmax = 1.1;
    else if (Theta_deg <= 0.7 && Theta_deg > 0.2)
        wmax = 2.;
    else if (Theta_deg <= 0.2 && Theta_deg > 0.05)
        wmax = 4.;
    else
        wmax = 8.;
    // We generate the q value according to the Jstarq pdf. We have to
    // define the qmin and qmax limits for this generation:
    K = sqrt(4. * T * (1. - IonizationEnergy_au / (2. * T) - sqrt(1. - IonizationEnergy_au / T) * c));
    Elmin = IonizationEnergy_au;
    Elmax = (anE + IonizationEnergy_eV) / 2. / 27.2;
    qmin = Elmin / K - K / 2.;
    qmax = Elmax / K - K / 2.;
    //
    q = qmax;
    Fmax = 1. / 2. + 1. / katrin::KConst::Pi() * (q / (1. + q * q) + atan(q));
    q = qmin;
    Fmin = 1. / 2. + 1. / katrin::KConst::Pi() * (q / (1. + q * q) + atan(q));
    h = Fmax - Fmin;
    // Generation of Eloss with the Neumann acceptance-rejection method:
    for (int j = 1; j < 5000; j++) {
        // Generation of q with inverse transform method
        // (we use the Newton-Raphson method in order to solve the nonlinear eq.
        // for the inversion) :
        RandomArray(3, u);
        //subrn(u,2);
        F = Fmin + h * u[1];
        y = 0.;
        for (int i = 1; i <= 30; i++) {
            G = 1. / 2. + (y + sin(2. * y) / 2.) / katrin::KConst::Pi();
            Gp = (1. + cos(2. * y)) / katrin::KConst::Pi();
            y = y - (G - F) / Gp;
            if (fabs(G - F) < 1.e-8)
                break;
        }
        q = tan(y);
        // We have the q value, so we can define El, and calculate the weight:
        El = q * K + K * K / 2.;
        // First Born approximation formula of Bethe for e-H ionization:
        KK = K;
        ki = sqrt(2. * T);
        kf = sqrt(2. * (T - El));
        K2 = 4. * T * (1. - El / (2. * T) - sqrt(1. - El / T) * c);
        if (K2 < 1.e-9)
            K2 = 1.e-9;
        K = sqrt(K2);  // momentum transfer
        Rex = 1. - K * K / (kf * kf) + K2 * K2 / (kf * kf * kf * kf);
        kej = sqrt(2. * fabs(El - IonizationEnergy_au) + 1.e-8);
        st1 = K2 - 2. * El + 2.;
        if (fabs(st1) < 1.e-9)
            st1 = 1.e-9;
        arg = 2. * kej / st1;
        if (arg >= 0.)
            arctg = atan(arg);
        else
            arctg = atan(arg) + katrin::KConst::Pi();
        st1 = (K + kej) * (K + kej) + 1.;
        st2 = (K - kej) * (K - kej) + 1.;
        fE = 1024. * El * (K2 + 2. / 3. * El) / (st1 * st1 * st1 * st2 * st2 * st2) * exp(-2. / kej * arctg) /
             (1. - exp(-2. * katrin::KConst::Pi() / kej));
        D2ion = 2. * kf / ki * Rex / (El * K2) * fE;
        K = KK;
        //
        WcE = D2ion;
        Jstarq = 16. / (3. * katrin::KConst::Pi() * (1. + q * q) * (1. + q * q));
        WcstarE = 4. / (K * K * K * K * K) * Jstarq;
        w = WcE / WcstarE;
        if (wmax * u[2] < w)
            break;
    }
    //
    Eloss = El * 27.21;
    if (Eloss < IonizationEnergy_eV + 0.05)
        Eloss = IonizationEnergy_eV + 0.05;
    //
    // << "Eloss " << Eloss << endl;

    return;

}  //end randomion

double InelasticFerencCalculator::DiffXSecExc(double anE, double cosTheta)
{
    double K2, K, T, theta;
    double sigma = 0.;
    //double a02=28.e-22;   // Bohr radius squared
    double a02 = katrin::KConst::BohrRadiusSquared();

    double EE = 12.6 / 27.21;
    double e[5] = {0., 25., 35., 50., 100.};
    double t[9] = {0., 10., 20., 30., 40., 60., 80., 100., 180.};
    double D[4][9] = {{60., 43., 27., 18., 13., 8., 6., 6., 6.},
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
    int i, j;
    //
    T = anE / 27.21;
    if (anE >= 100.) {
        K2 = 4. * T * (1. - EE / (2. * T) - sqrt(1. - EE / T) * cosTheta);
        if (K2 < 1.e-9)
            K2 = 1.e-9;
        K = sqrt(K2);  // momentum transfer
        sigma = 2. / K2 * sumexc(K) * a02;
    }
    else if (anE <= 10.)
        sigma = 0.;
    else {
        theta = acos(cosTheta) * 180. / katrin::KConst::Pi();
        for (i = 0; i <= 3; i++)
            if (anE >= e[i] && anE < e[i + 1])
                for (j = 0; j <= 7; j++)
                    if (theta >= t[j] && theta < t[j + 1])
                        sigma = 1.e-22 * (D[i][j] + (D[i][j + 1] - D[i][j]) * (theta - t[j]) / (t[j + 1] - t[j]));
    }
    return sigma;

}  //end DiffXSecExc

double InelasticFerencCalculator::DiffXSecInel(double anE, double cosTheta)
{

    //double a02=28.e-22;   // Bohr radius squared
    double a02 = katrin::KConst::BohrRadiusSquared();

    double Cinel[50] = {-0.246, -0.244, -0.239, -0.234, -0.227, -0.219, -0.211, -0.201, -0.190, -0.179,
                        -0.167, -0.155, -0.142, -0.130, -0.118, -0.107, -0.096, -0.085, -0.076, -0.067,
                        -0.059, -0.051, -0.045, -0.039, -0.034, -0.029, -0.025, -0.022, -0.019, -0.016,
                        -0.014, -0.010, -0.008, -0.006, -0.004, -0.003, -0.003, -0.002, -0.002, -0.001,
                        -0.001, -0.001, 0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000};
    //double Ei=0.568;
    double IonizationEnergy_eV = GetIonizationEnergy();        //ionization energy in eV
    double IonizationEnergy_au = IonizationEnergy_eV / 27.21;  //ionization energy in atomic units
    double T, K2, K, st1, F, DH, Dinelreturn, CinelK, Ki;
    int i;
    if (anE < IonizationEnergy_eV)  //Achtung! 16.
        return DiffXSecExc(anE, cosTheta);
    T = anE / 27.21;
    K2 = 4. * T * (1. - IonizationEnergy_au / (2. * T) - sqrt(1. - IonizationEnergy_au / T) * cosTheta);
    if (K2 < 1.e-9)
        K2 = 1.e-9;  //Achtung!!
    K = sqrt(K2);    // momentum transfer
    st1 = 1. + K2 / 4.;
    F = 1. / (st1 * st1);  // scatt. formfactor of hydrogen atom
    // DH is the diff. cross section for inelastic electron scatt.
    // on atomic hydrogen within the first Born approximation :
    DH = 4. / (K2 * K2) * (1. - F * F) * a02;
    // CinelK calculation with linear interpolation.
    // CinelK is the correction of the inelastic electron
    // scatt. on molecular hydrogen compared to the independent atom
    // model.
    if (K < 3.) {
        i = (int) (K / 0.1);  //WOLF: double->int
        Ki = i * 0.1;         //WOLF: double->int
        CinelK = Cinel[i] + (K - Ki) / 0.1 * (Cinel[i + 1] - Cinel[i]);
    }
    else if (K >= 3. && K < 5.) {
        i = (int) (30 + (K - 3.) / 0.2);  //WOLF: Double->int
        Ki = 3. + (i - 30) * 0.2;         //WOLF double->int
        CinelK = Cinel[i] + (K - Ki) / 0.2 * (Cinel[i + 1] - Cinel[i]);
    }
    else if (K >= 5. && K < 9.49) {
        i = (int) (40 + (K - 5.) / 0.5);  //WOLF: double ->int
        Ki = 5. + (i - 40) * 0.5;         //WOLF: double->Int
        CinelK = Cinel[i] + (K - Ki) / 0.5 * (Cinel[i + 1] - Cinel[i]);
    }
    else
        CinelK = 0.;
    Dinelreturn = 2. * DH * (1. + CinelK);
    return Dinelreturn;
}

void InelasticFerencCalculator::gensecelen(double anE, double& W)
{  //TODO

    double IonizationEnergy_eV = GetIonizationEnergy();  //ionization energy in eV
    double /*Ei=15.45,*/ eps2 = 14.3, b = 6.25;
    double B;
    double C, A, eps, a, u, epsmax;

    B = atan((IonizationEnergy_eV - eps2) / b);
    epsmax = (anE + IonizationEnergy_eV) / 2.;
    A = atan((epsmax - eps2) / b);
    C = b / (A - B);
    u = KRandom::GetInstance().Uniform();
    a = b / C * (u + C / b * B);
    eps = eps2 + b * tan(a);
    W = eps - IonizationEnergy_eV;
    return;
}

double InelasticFerencCalculator::sigmainel(double anE)
{

    if (anE < 250.) {
        intmsg(eWarning) << "InelasticFerencCalculator::sigmainel" << ret;
        intmsg << "cross section not valid. return 0." << eom;
        return 0.0;
    }

    double IonizationEnergy_eV = GetIonizationEnergy();
    double IonizationEnergy_au = IonizationEnergy_eV / 27.21;  //ionization energy in atomic units
    //double Ei=0.568;  // ionization energy of molecular
    // hydrogen in Hartree atomic units
    //  (15.45 eV)
    double a02 = katrin::KConst::BohrRadiusSquared();

    double sigma, gamtot, T;
    T = anE / 27.21;
    gamtot = 2. * (-7. / 4. + log(IonizationEnergy_au / (2. * T)));
    sigma = 2. * katrin::KConst::Pi() / T * (1.5487 * log(2. * T) + 2.4036 + gamtot / (2. * T));
    sigma = sigma * a02;
    return sigma;
}

/////////////////////////////////////////////////////////////
//private helper methods

double InelasticFerencCalculator::sumexc(double K)
{
    double Kvec[15] = {0., 0.1, 0.2, 0.4, 0.6, 0.8, 1., 1.2, 1.5, 1.8, 2., 2.5, 3., 4., 5.};
    double fvec[7][15] = {{2.907e-1,
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
    double EeV[7] = {12.73, 13.20, 14.77, 15.3, 14.93, 15.4, 13.06};
    int jmin = 0;
    int nnmax = 6;
    double En, f[7], x4[4], f4[4], sum;
    //
    sum = 0.;

    for (int n = 0; n <= nnmax; n++) {
        En = EeV[n] / 27.21;  // En is the excitation energy in Hartree atomic units
        if (K >= 5.)
            f[n] = 0.;
        else if (K >= 3. && K <= 4.)
            f[n] = fvec[n][12] + (K - 3.) * (fvec[n][13] - fvec[n][12]);
        else if (K >= 4. && K <= 5.)
            f[n] = fvec[n][13] + (K - 4.) * (fvec[n][14] - fvec[n][13]);
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
                x4[j] = Kvec[jmin + j];
                f4[j] = fvec[n][jmin + j];
            }
            f[n] = Lagrange(4, x4, f4, K);
        }
        sum += f[n] / En;
    }
    return sum;
}  //end sumexc

double InelasticFerencCalculator::sigmadiss10(double anE)
{

    double a[9] = {-2.297914361e5,
                   5.303988579e5,
                   -5.316636672e5,
                   3.022690779e5,
                   -1.066224144e5,
                   2.389841369e4,
                   -3.324526406e3,
                   2.624761592e2,
                   -9.006246604};
    double lnsigma, lnE, lnEn, Emin, sigma;
    int n;
    //  anE is in eV
    sigma = 0.;
    Emin = 9.8;
    lnE = log(anE);
    lnEn = 1.;
    lnsigma = 0.;
    if (anE < Emin)
        sigma = 0.;
    else {
        for (n = 0; n <= 8; n++) {
            lnsigma += a[n] * lnEn;
            lnEn = lnEn * lnE;
        }
        sigma = exp(lnsigma);
    }
    return sigma * 1.e-4;
}

double InelasticFerencCalculator::sigmadiss15(double anE)
{

    double a[9] = {-1.157041752e3,
                   1.501936271e3,
                   -8.6119387e2,
                   2.754926257e2,
                   -5.380465012e1,
                   6.573972423,
                   -4.912318139e-1,
                   2.054926773e-2,
                   -3.689035889e-4};
    double lnsigma, lnE, lnEn, Emin, sigma;
    int n;
    //  anE is in eV
    sigma = 0.;
    Emin = 16.5;
    lnE = log(anE);
    lnEn = 1.;
    lnsigma = 0.;
    if (anE < Emin)
        sigma = 0.;
    else {
        for (n = 0; n <= 8; n++) {
            lnsigma += a[n] * lnEn;
            lnEn = lnEn * lnE;
        }
        sigma = exp(lnsigma);
    }
    return sigma * 1.e-4;
}

double InelasticFerencCalculator::sigmaBC(double anE)
{

    double aB[9] = {-4.2935194e2,
                    5.1122109e2,
                    -2.8481279e2,
                    8.8310338e1,
                    -1.6659591e1,
                    1.9579609,
                    -1.4012824e-1,
                    5.5911348e-3,
                    -9.5370103e-5};
    double aC[9] = {-8.1942684e2,
                    9.8705099e2,
                    -5.3095543e2,
                    1.5917023e2,
                    -2.9121036e1,
                    3.3321027,
                    -2.3305961e-1,
                    9.1191781e-3,
                    -1.5298950e-4};
    double lnsigma, lnE, lnEn, sigmaB, Emin, sigma, sigmaC;
    int n;
    sigma = 0.;
    Emin = 12.5;
    lnE = log(anE);
    lnEn = 1.;
    lnsigma = 0.;
    if (anE < Emin)
        sigmaB = 0.;
    else {
        for (n = 0; n <= 8; n++) {
            lnsigma += aB[n] * lnEn;
            lnEn = lnEn * lnE;
        }
        sigmaB = exp(lnsigma);
    }
    sigma += sigmaB;
    //  sigma=0.;
    // C state:
    Emin = 15.8;
    lnE = log(anE);
    lnEn = 1.;
    lnsigma = 0.;
    if (anE < Emin)
        sigmaC = 0.;
    else {
        for (n = 0; n <= 8; n++) {
            lnsigma += aC[n] * lnEn;
            lnEn = lnEn * lnE;
        }
        sigmaC = exp(lnsigma);
    }
    sigma += sigmaC;
    return sigma * 1.e-4;
}
}  // namespace Kassiopeia

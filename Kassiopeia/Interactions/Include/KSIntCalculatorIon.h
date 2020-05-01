#ifndef Kassiopeia_KSIntCalculatorIon_h_
#define Kassiopeia_KSIntCalculatorIon_h_

#include "KField.h"
#include "KSIntCalculator.h"
#include "KSIntCalculatorHydrogen.h"

namespace Kassiopeia
{
class KSIntCalculatorIon : public KSComponentTemplate<KSIntCalculatorIon, KSIntCalculator>
{
  public:
    KSIntCalculatorIon();
    KSIntCalculatorIon(const KSIntCalculatorIon& aCopy);
    KSIntCalculatorIon* Clone() const override;
    ~KSIntCalculatorIon() override;

  public:
    void CalculateCrossSection(const KSParticle& aParticle, double& aCrossSection) override;
    void ExecuteInteraction(const KSParticle& anIncomingIon, KSParticle& anOutgoingIon,
                            KSParticleQueue& aSecondaries) override;
    void CalculateEnergyDifferentialCrossSection(const double anIncomingIonMass, const double anIncomingIonEnergy,
                                                 const double aSecondaryElectronEnergy, double& aCrossSection);
    void CalculateAngleDifferentialCrossSection(const double aSecondaryElectronAngle, double& aCrossSection);

  protected:
    double Hplus_H2_crossSection(double aEnergy);
    double H2plus_H2_crossSection(double aEnergy);
    double H3plus_H2_crossSection(double aEnergy);

    double f1(double x, double c1, double c2);
    double f2(double x, double c1, double c2, double c3, double c4);
    double f3(double x, double c1, double c2, double c3, double c4, double c5, double c6);
    double sigma1(double E1, double a1, double a2, double a3, double a4);
    double sigma2(double E1, double a1, double a2, double a3, double a4, double a5, double a6);
    double sigma6(double E1, double a1, double a2, double a3, double a4, double a5, double a6);
    double sigma10(double E1, double a1, double a2, double a3, double a4, double a5, double a6, double a7, double a8);
    double E_1(double E, double E_threshold);

    double Hplus_H2O_crossSection(double aEnergy);

    double sigmatot(double aEnergy, double A, double B, double C, double D);
    double sigmalow(double x, double C, double D);
    double sigmahigh(double x, double A, double B);

    void CalculateSecondaryElectronEnergy(const double anIncomingIonMass, const double anIncomingIonEnergy,
                                          double& aSecondaryElectronEnergy);
    void CalculateSecondaryElectronAngle(double& aSecondaryElectronEnergy);
    //For debugging purposes, moved to public:
    //void CalculateEnergyDifferentialCrossSection( const double anIncomingIonMass, const double anIncomingIonEnergy,const double aSecondaryElectronEnergy,double &aCrossSection);

    double F_1(double v, double A_1, double B_1, double C_1, double D_1, double E_1);
    double F_2(double v, double A_2, double B_2, double C_2, double D_2);
    double H_1(double v, double A_1, double B_1);
    double H_2(double v, double A_2, double B_2);
    double L_1(double v, double C_1, double D_1, double E_1);
    double L_2(double v, double C_2, double D_2);
    double w_c(double v, double I);

  public:
    K_SET_GET(std::string, Gas);

  protected:
    double E_Binding;
};

}  // namespace Kassiopeia

#endif

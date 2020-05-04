//
// Created by wdconinc on 13.02.20.
//

#ifndef KASPER_KSGENENERGYBETARECOIL_H
#define KASPER_KSGENENERGYBETARECOIL_H

#include "KField.h"
#include "KSGenCreator.h"

namespace Kassiopeia
{
class KSGenEnergyBetaRecoil : public KSComponentTemplate<KSGenEnergyBetaRecoil, KSGenCreator>
{
  public:
    KSGenEnergyBetaRecoil();
    KSGenEnergyBetaRecoil(const KSGenEnergyBetaRecoil& aCopy);
    KSGenEnergyBetaRecoil* Clone() const;
    virtual ~KSGenEnergyBetaRecoil();

    //******
    //action
    //******

  public:
    void Dice(KSParticleQueue* aPrimaries);

    //*************
    //configuration
    //*************

  public:
    double g(double E);
    double g1(double E);
    double g2(double E);
    double GetRecoilEnergyMax();
    double GetRecoilEnergyProbabilityMax(double Emax);
    double GenRecoilEnergy();

  private:
    ;
    K_SET_GET(int, NMax);
    ;
    K_SET_GET(double, EMax);
    ;
    K_SET_GET(double, PMax);
    ;
    K_SET_GET(double, MinEnergy);
    ;
    K_SET_GET(double, MaxEnergy);

    //**********
    //initialize
    //**********

  public:
    void InitializeComponent();
    void DeinitializeComponent();
};

}  // namespace Kassiopeia

#endif  //KASPER_KSGENENERGYBETARECOIL_H

#ifndef Kassiopeia_KSGenEnergyRadonEvent_h_
#define Kassiopeia_KSGenEnergyRadonEvent_h_

#include "KSGenCreator.h"

namespace Kassiopeia
{
class KSGenShakeOff;
class KSGenRelaxation;
class KSGenConversion;

class KSGenEnergyRadonEvent : public KSComponentTemplate<KSGenEnergyRadonEvent, KSGenCreator>
{
  public:
    KSGenEnergyRadonEvent();
    KSGenEnergyRadonEvent(const KSGenEnergyRadonEvent& aCopy);
    KSGenEnergyRadonEvent* Clone() const override;
    ~KSGenEnergyRadonEvent() override;

    //******
    //action
    //******

  public:
    void Dice(KSParticleQueue* aPrimaries) override;

    //*************
    //configuration
    //*************

  public:
    void SetForceConversion(bool aSetting);
    void SetForceShakeOff(bool aSetting);
    void SetDoConversion(bool aSetting);
    void SetDoShakeOff(bool aSetting);
    void SetDoAuger(bool aSetting);
    void SetIsotope(int anIsotope);

  private:
    bool fForceConversion;
    bool fForceShakeOff;
    bool fDoConversion;
    bool fDoShakeOff;
    bool fDoAuger;
    int fIsotope;

    //**********
    //initialize
    //**********

  public:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    KSGenRelaxation* fMyRelaxation;
    KSGenShakeOff* fMyShakeOff;
    KSGenConversion* fMyConversion;
};

}  // namespace Kassiopeia

#endif

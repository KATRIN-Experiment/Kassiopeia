#ifndef Kassiopeia_KSGenEnergyKryptonEvent_h_
#define Kassiopeia_KSGenEnergyKryptonEvent_h_

#include "KSGenCreator.h"

namespace Kassiopeia
{
class KSGenRelaxation;
class KSGenConversion;

class KSGenEnergyKryptonEvent : public KSComponentTemplate<KSGenEnergyKryptonEvent, KSGenCreator>
{
  public:
    KSGenEnergyKryptonEvent();
    KSGenEnergyKryptonEvent(const KSGenEnergyKryptonEvent& aCopy);
    KSGenEnergyKryptonEvent* Clone() const override;
    ~KSGenEnergyKryptonEvent() override;

  public:
    void Dice(KSParticleQueue* aPrimaries) override;

    //*************
    //configuration
    //*************

  public:
    void SetForceConversion(bool aSetting);
    void SetDoConversion(bool aSetting);
    void SetDoAuger(bool aSetting);


  private:
    bool fForceConversion;
    bool fDoConversion;
    bool fDoAuger;

    //**************
    //initialization
    //**************

  public:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    KSGenRelaxation* fMyRelaxation;
    KSGenConversion* fMyConversion;
};

}  // namespace Kassiopeia

#endif

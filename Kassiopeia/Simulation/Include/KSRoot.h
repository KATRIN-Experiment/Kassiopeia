#ifndef Kassiopeia_KSRoot_h_
#define Kassiopeia_KSRoot_h_

#include "KSComponentTemplate.h"
#include "KSMainMessage.h"
#include "KToolbox.h"

namespace Kassiopeia
{
class KSRootMagneticField;
class KSRootElectricField;
class KSRootSpace;
class KSRootGenerator;
class KSRootTrajectory;
class KSRootSpaceInteraction;
class KSRootSpaceNavigator;
class KSRootSurfaceInteraction;
class KSRootSurfaceNavigator;
class KSRootTerminator;
class KSRootWriter;
class KSRootStepModifier;
class KSRootTrackModifier;
class KSRootEventModifier;
class KSRootRunModifier;

class KSSimulation;
class KSRun;
class KSEvent;
class KSTrack;
class KSStep;

class KSRoot : public KSComponentTemplate<KSRoot>
{
  public:
    KSRoot();
    KSRoot(const KSRoot& aCopy);
    KSRoot* Clone() const override;
    ~KSRoot() override;

  public:
    void Execute(KSSimulation* aSimulation);

  protected:
    void ExecuteRun();
    void ExecuteEvent();
    void ExecuteTrack();
    void ExecuteStep();

  protected:
    void ActivateComponent() override;
    void DeactivateComponent() override;
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    static void SignalHandler(int aSignal);

  private:
    KSSimulation* fSimulation;
    KSRun* fRun;
    KSEvent* fEvent;
    KSTrack* fTrack;
    KSStep* fStep;

    KSRootMagneticField* fRootMagneticField;
    KSRootElectricField* fRootElectricField;
    KSRootSpace* fRootSpace;
    KSRootGenerator* fRootGenerator;
    KSRootTrajectory* fRootTrajectory;
    KSRootSpaceInteraction* fRootSpaceInteraction;
    KSRootSpaceNavigator* fRootSpaceNavigator;
    KSRootSurfaceInteraction* fRootSurfaceInteraction;
    KSRootSurfaceNavigator* fRootSurfaceNavigator;
    KSRootTerminator* fRootTerminator;
    KSRootWriter* fRootWriter;
    KSRootStepModifier* fRootStepModifier;
    KSRootTrackModifier* fRootTrackModifier;
    KSRootEventModifier* fRootEventModifier;
    KSRootRunModifier* fRootRunModifier;

    bool fOnce;
    bool fRestartNavigation;

    unsigned int fRunIndex;
    unsigned int fEventIndex;
    unsigned int fTrackIndex;
    unsigned int fStepIndex;

    double fTotalExecTime;

    static bool fStopRunSignal;
    static bool fStopEventSignal;
    static bool fStopTrackSignal;
};

}  // namespace Kassiopeia

#endif

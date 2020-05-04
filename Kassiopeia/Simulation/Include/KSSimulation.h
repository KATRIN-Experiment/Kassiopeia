#ifndef Kassiopeia_KSSimulation_h_
#define Kassiopeia_KSSimulation_h_

#include "KSComponentTemplate.h"

#include <vector>


namespace Kassiopeia
{
class KSRunModifier;
class KSEventModifier;
class KSTrackModifier;
class KSStepModifier;

class KSSimulation : public KSComponentTemplate<KSSimulation>
{
  public:
    KSSimulation();
    KSSimulation(const KSSimulation& aCopy);
    KSSimulation* Clone() const override;
    ~KSSimulation() override;

  public:
    void SetSeed(const unsigned int& aSeed);
    const unsigned int& GetSeed() const;

    void SetRun(const unsigned int& aRun);
    const unsigned int& GetRun() const;

    void SetEvents(const unsigned int& anEvents);
    const unsigned int& GetEvents() const;

    void SetStepReportIteration(const unsigned int& anIteration);
    const unsigned int& GetStepReportIteration() const;

    void AddCommand(KSCommand* aCommand);
    void RemoveCommand(KSCommand* aCommand);

    //static modifiers, which are always present regardless of simulation state
    void AddStaticRunModifier(KSRunModifier* runModifier)
    {
        fStaticRunModifiers.push_back(runModifier);
    };
    void AddStaticEventModifier(KSEventModifier* eventModifier)
    {
        fStaticEventModifiers.push_back(eventModifier);
    };
    void AddStaticTrackModifier(KSTrackModifier* trackModifier)
    {
        fStaticTrackModifiers.push_back(trackModifier);
    };
    void AddStaticStepModifier(KSStepModifier* stepModifier)
    {
        fStaticStepModifiers.push_back(stepModifier);
    };

    std::vector<KSRunModifier*>* GetStaticRunModifiers()
    {
        return &fStaticRunModifiers;
    };
    std::vector<KSEventModifier*>* GetStaticEventModifiers()
    {
        return &fStaticEventModifiers;
    };
    std::vector<KSTrackModifier*>* GetStaticTrackModifiers()
    {
        return &fStaticTrackModifiers;
    };
    std::vector<KSStepModifier*>* GetStaticStepModifiers()
    {
        return &fStaticStepModifiers;
    };

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;
    void ActivateComponent() override;
    void DeactivateComponent() override;

    unsigned int fSeed;
    unsigned int fRun;
    unsigned int fEvents;
    unsigned int fStepReportIteration;
    std::vector<KSCommand*> fCommands;
    std::vector<KSRunModifier*> fStaticRunModifiers;
    std::vector<KSEventModifier*> fStaticEventModifiers;
    std::vector<KSTrackModifier*> fStaticTrackModifiers;
    std::vector<KSStepModifier*> fStaticStepModifiers;
};

}  // namespace Kassiopeia

#endif

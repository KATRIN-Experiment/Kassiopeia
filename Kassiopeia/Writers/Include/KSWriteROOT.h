#ifndef Kassiopeia_KSWriteROOT_h_
#define Kassiopeia_KSWriteROOT_h_

#include "KFile.h"
#include "KRootFile.h"
#include "KSList.h"
#include "KSWriteROOTCondition.h"
#include "KSWriter.h"
#include "KThreeVector.hh"
#include "KTwoVector.hh"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TVector3.h"

#include <map>

namespace Kassiopeia
{

class KSWriteROOT : public KSComponentTemplate<KSWriteROOT, KSWriter>
{
  private:
    class Data
    {
      public:
        Data(KSComponent* aComponent);
        ~Data();

        void Start(const unsigned int& anIndex);
        void Fill();
        void Stop();

      private:
        void MakeTrees(KSComponent* aComponent);
        void MakeBranches(KSComponent* aComponent);

        TTree* fStructure;
        std::string fLabel;
        std::string fType;

        TTree* fPresence;
        unsigned int fIndex;
        unsigned int fLength;

        TTree* fData;

        std::vector<KSComponent*> fComponents;
    };

    using KSComponentMap = std::map<KSComponent*, Data*>;
    using ComponentIt = KSComponentMap::iterator;
    using ComponentCIt = KSComponentMap::const_iterator;
    using ComponentEntry = KSComponentMap::value_type;

  public:
    KSWriteROOT();
    KSWriteROOT(const KSWriteROOT& aCopy);
    KSWriteROOT* Clone() const override;
    ~KSWriteROOT() override;

  public:
    void SetBase(const std::string& aBase);
    void SetPath(const std::string& aPath);
    void SetStepIteration(const unsigned int& aValue);

  private:
    std::string fBase;
    std::string fPath;
    unsigned int fStepIteration;
    unsigned int fStepIterationIndex;

  public:
    void AddRunWriteCondition(KSWriteROOTCondition* aWriteCondition);
    void RemoveRunWriteCondition(KSWriteROOTCondition* aWriteCondition);

    void AddEventWriteCondition(KSWriteROOTCondition* aWriteCondition);
    void RemoveEventWriteCondition(KSWriteROOTCondition* aWriteCondition);

    void AddTrackWriteCondition(KSWriteROOTCondition* aWriteCondition);
    void RemoveTrackWriteCondition(KSWriteROOTCondition* aWriteCondition);

    void AddStepWriteCondition(KSWriteROOTCondition* aWriteCondition);
    void RemoveStepWriteCondition(KSWriteROOTCondition* aWriteCondition);

  private:
    KSList<KSWriteROOTCondition> fRunWriteConditions;
    KSList<KSWriteROOTCondition> fEventWriteConditions;
    KSList<KSWriteROOTCondition> fTrackWriteConditions;
    KSList<KSWriteROOTCondition> fStepWriteConditions;

  public:
    void ExecuteRun() override;
    void ExecuteEvent() override;
    void ExecuteTrack() override;
    void ExecuteStep() override;

    void AddRunComponent(KSComponent* aComponent);
    void RemoveRunComponent(KSComponent* aComponent);

    void AddEventComponent(KSComponent* aComponent);
    void RemoveEventComponent(KSComponent* aComponent);

    void AddTrackComponent(KSComponent* aComponent);
    void RemoveTrackComponent(KSComponent* aComponent);

    void AddStepComponent(KSComponent* aComponent);
    void RemoveStepComponent(KSComponent* aComponent);

  protected:
    void InitializeComponent() override;
    void DeinitializeComponent() override;

  private:
    katrin::KRootFile* fFile;
    std::string fKey;

    TTree* fRunKeys;
    TTree* fRunData;
    KSComponentMap fRunComponents;
    KSComponentMap fActiveRunComponents;
    unsigned int fRunIndex;
    unsigned int fRunFirstEventIndex;
    unsigned int fRunLastEventIndex;
    unsigned int fRunFirstTrackIndex;
    unsigned int fRunLastTrackIndex;
    unsigned int fRunFirstStepIndex;
    unsigned int fRunLastStepIndex;

    TTree* fEventKeys;
    TTree* fEventData;
    KSComponentMap fEventComponents;
    KSComponentMap fActiveEventComponents;
    unsigned int fEventIndex;
    unsigned int fEventFirstTrackIndex;
    unsigned int fEventLastTrackIndex;
    unsigned int fEventFirstStepIndex;
    unsigned int fEventLastStepIndex;

    TTree* fTrackKeys;
    TTree* fTrackData;
    KSComponentMap fTrackComponents;
    KSComponentMap fActiveTrackComponents;
    unsigned int fTrackIndex;
    unsigned int fTrackFirstStepIndex;
    unsigned int fTrackLastStepIndex;

    bool fStepComponent;
    TTree* fStepKeys;
    TTree* fStepData;
    KSComponentMap fStepComponents;
    KSComponentMap fActiveStepComponents;
    unsigned int fStepIndex;

    static const Int_t fBufferSize;
    static const Int_t fSplitLevel;
    static const std::string fLabel;
};

inline void KSWriteROOT::SetBase(const std::string& aBase)
{
    fBase = aBase;
    return;
}
inline void KSWriteROOT::SetPath(const std::string& aPath)
{
    fPath = aPath;
    return;
}
inline void KSWriteROOT::SetStepIteration(const unsigned int& aValue)
{
    fStepIteration = aValue;
    return;
}

}  // namespace Kassiopeia

#endif

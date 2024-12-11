#ifndef Kassiopeia_KSWriteASCII_h_
#define Kassiopeia_KSWriteASCII_h_

#include "KFile.h"
#include "KSWriter.h"
#include "KTextFile.h"

#include <map>

namespace Kassiopeia
{

class KSWriteASCII : public KSComponentTemplate<KSWriteASCII, KSWriter>
{
  private:
    class Data
    {
      public:
        Data(KSComponent* aComponent, int aPrecision);
        ~Data();

        void Initialize(KSComponent* aComponent, int aPrecision);
        void Start(const unsigned int& anIndex);
        std::string ValuesAsString();
        std::string Label();


      private:
        std::string fLabel;
        std::string fType;
        unsigned int fIndex;
        unsigned int fLength;

        class OutputObjectASCII
        {
          private:
            KSComponent* fComponent;
            std::string fType;
            int fPrecision;

          public:
            OutputObjectASCII(KSComponent* aComponent, std::string aType, int Precision);
            ~OutputObjectASCII();
            std::string getValue();
        };

        std::vector<KSComponent*> fComponents;
        std::vector<OutputObjectASCII*> fOutputObjectASCIIs;
    };

    using KSComponentMap = std::map<KSComponent*, Data*>;
    using ComponentIt = KSComponentMap::iterator;
    using ComponentCIt = KSComponentMap::const_iterator;
    using ComponentEntry = KSComponentMap::value_type;

  public:
    KSWriteASCII();
    KSWriteASCII(const KSWriteASCII& aCopy);
    KSWriteASCII* Clone() const override;
    ~KSWriteASCII() override;

  public:
    void SetBase(const std::string& aBase);
    void SetPath(const std::string& aPath);
    void SetStepIteration(const unsigned int& aValue);
    void SetPrecision(const unsigned int& aValue);

    katrin::KTextFile* TextFile();
    void Write(std::string str);
    void Write(char c);
    int Precision() const;

  protected:
    void MakeOutputFile(int anIndex);

  private:
    std::string fBase;
    std::string fPath;
    unsigned int fStepIteration;
    unsigned int fStepIterationIndex;
    unsigned int fPrecision;

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
    katrin::KTextFile* fTextFile;

    std::string fKey;
    KSComponentMap fRunComponents;
    KSComponentMap fActiveRunComponents;
    unsigned int fRunIndex;
    unsigned int fRunFirstEventIndex;
    unsigned int fRunLastEventIndex;
    unsigned int fRunFirstTrackIndex;
    unsigned int fRunLastTrackIndex;
    unsigned int fRunFirstStepIndex;
    unsigned int fRunLastStepIndex;

    KSComponentMap fEventComponents;
    KSComponentMap fActiveEventComponents;
    unsigned int fEventIndex;
    unsigned int fEventFirstTrackIndex;
    unsigned int fEventLastTrackIndex;
    unsigned int fEventFirstStepIndex;
    unsigned int fEventLastStepIndex;

    KSComponentMap fTrackComponents;
    KSComponentMap fActiveTrackComponents;
    unsigned int fTrackIndex;
    unsigned int fTrackFirstStepIndex;
    unsigned int fTrackLastStepIndex;

    bool fStepComponent;
    KSComponentMap fStepComponents;
    KSComponentMap fActiveStepComponents;
    unsigned int fStepIndex;

    static const int fBufferSize;
    static const int fSplitLevel;
    static const std::string fLabel;
};

inline void KSWriteASCII::SetBase(const std::string& aBase)
{
    fBase = aBase;
    return;
}
inline void KSWriteASCII::SetPath(const std::string& aPath)
{
    fPath = aPath;
    return;
}
inline void KSWriteASCII::SetStepIteration(const unsigned int& aValue)
{
    fStepIteration = aValue;
    return;
}

inline void KSWriteASCII::SetPrecision(const unsigned int& aValue)
{
    fPrecision = aValue;
    return;
}

inline katrin::KTextFile* KSWriteASCII::TextFile()
{
    if (!fTextFile)
        MakeOutputFile(fTrackIndex);
    
    return fTextFile;
}
    
inline void KSWriteASCII::Write(std::string str)
{
    for (char& it : str)
        TextFile()->File()->put(it);
}
    
inline void KSWriteASCII::Write(char c)
{
    TextFile()->File()->put(c);
}

inline int KSWriteASCII::Precision() const
{
    return fPrecision;
}
    
inline std::string KSWriteASCII::Data::Label()
{
    return fLabel;
}

}  // namespace Kassiopeia


#endif

#ifndef _Kassiopeia_KSReadObjectROOT_h_
#define _Kassiopeia_KSReadObjectROOT_h_

#include "KSReadIterator.h"
#include "TFile.h"
#include "TTree.h"

namespace Kassiopeia
{
class KSReadObjectROOT :
    public KSReadIterator,
    public KSBoolSet,
    public KSUCharSet,
    public KSCharSet,
    public KSUShortSet,
    public KSShortSet,
    public KSUIntSet,
    public KSIntSet,
    public KSULongSet,
    public KSLongSet,
    public KSLongLongSet,
    public KSFloatSet,
    public KSDoubleSet,
    public KSThreeVectorSet,
    public KSTwoVectorSet,
    public KSStringSet
{
  public:
    using KSReadIterator::Add;
    using KSReadIterator::Exists;
    using KSReadIterator::Get;

  public:
    KSReadObjectROOT(TTree* aStructureTree, TTree* aPresenceTree, TTree* aDataTree);
    ~KSReadObjectROOT() override;

  public:
    void operator++(int) override;
    void operator--(int) override;
    void operator<<(const unsigned int& aValue) override;

  public:
    bool Valid() const override;
    unsigned int Index() const override;
    bool operator<(const unsigned int& aValue) const override;
    bool operator<=(const unsigned int& aValue) const override;
    bool operator>(const unsigned int& aValue) const override;
    bool operator>=(const unsigned int& aValue) const override;
    bool operator==(const unsigned int& aValue) const override;
    bool operator!=(const unsigned int& aValue) const override;

  private:
    class Presence
    {
      public:
        Presence(unsigned int anIndex, unsigned int aLength, unsigned int anEntry) :
            fIndex(anIndex),
            fLength(aLength),
            fEntry(anEntry)
        {}

        unsigned int fIndex;
        unsigned int fLength;
        unsigned int fEntry;
    };

    std::vector<Presence> fPresences;
    bool fValid;
    unsigned int fIndex;
    TTree* fStructure;
    TTree* fPresence;
    TTree* fData;
};

}  // namespace Kassiopeia

#endif

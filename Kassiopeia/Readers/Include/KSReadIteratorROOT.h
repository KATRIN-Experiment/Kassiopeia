#ifndef _Kassiopeia_KSReadIteratorROOT_h_
#define _Kassiopeia_KSReadIteratorROOT_h_

#include "KSReadObjectROOT.h"

namespace Kassiopeia
{
class KSReadIteratorROOT :
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
  private:
    using ObjectMap = std::map<std::string, KSReadObjectROOT*>;
    using ObjectIt = ObjectMap::iterator;
    using ObjectCIt = ObjectMap::const_iterator;
    using ObjectEntry = ObjectMap::value_type;

  public:
    KSReadIteratorROOT(TFile* aFile, TTree* aKeyTree, TTree* aDataTree);
    ~KSReadIteratorROOT() override;

    //*********
    //traversal
    //*********

  public:
    void operator<<(const unsigned int& aValue) override;
    void operator++(int) override;
    void operator--(int) override;

    //**********
    //comparison
    //**********

  public:
    bool Valid() const override;
    unsigned int Index() const override;
    bool operator<(const unsigned int& aValue) const override;
    bool operator<=(const unsigned int& aValue) const override;
    bool operator>(const unsigned int& aValue) const override;
    bool operator>=(const unsigned int& aValue) const override;
    bool operator==(const unsigned int& aValue) const override;
    bool operator!=(const unsigned int& aValue) const override;

    //*******
    //content
    //*******

  public:
    bool HasObject(const std::string& aLabel);
    KSReadObjectROOT& GetObject(const std::string& aLabel);
    const KSReadObjectROOT& GetObject(const std::string& aLabel) const;

  protected:
    TTree* fData;
    bool fValid;
    unsigned int fIndex;
    unsigned int fFirstIndex;
    unsigned int fLastIndex;
    ObjectMap fObjects;
};


}  // namespace Kassiopeia

#endif

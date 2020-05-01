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
    public KSFloatSet,
    public KSDoubleSet,
    public KSThreeVectorSet,
    public KSTwoVectorSet,
    public KSStringSet
{
  private:
    typedef map<std::string, KSReadObjectROOT*> ObjectMap;
    typedef ObjectMap::iterator ObjectIt;
    typedef ObjectMap::const_iterator ObjectCIt;
    typedef ObjectMap::value_type ObjectEntry;

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

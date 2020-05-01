#ifndef KFMNamedScalarDataCollection_HH__
#define KFMNamedScalarDataCollection_HH__

#include "KFMNamedScalarData.hh"
#include "KSAStructuredASCIIHeaders.hh"

#include <string>
#include <vector>

namespace KEMField
{

/*
*
*@file KFMNamedScalarDataCollection.hh
*@class KFMNamedScalarDataCollection
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Apr 23 10:38:33 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMNamedScalarDataCollection : public KSAInputOutputObject
{
  public:
    KFMNamedScalarDataCollection()
    {
        fCollectionName = "data";
    };
    ~KFMNamedScalarDataCollection() override{};

    const KFMNamedScalarData* GetDataWithName(std::string name) const;
    KFMNamedScalarData* GetDataWithName(std::string name);
    void AddData(const KFMNamedScalarData& data);

    unsigned int GetNDataSets() const
    {
        return fData.size();
    };
    const KFMNamedScalarData* GetDataSetWithIndex(unsigned int i) const
    {
        return &(fData[i]);
    };
    KFMNamedScalarData* GetDataSetWithIndex(unsigned int i)
    {
        return &(fData[i]);
    };

    std::string GetCollectionName() const
    {
        return fCollectionName;
    };
    void SetCollectionName(const std::string& name)
    {
        fCollectionName = name;
    };


    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;
    virtual const char* ClassName() const
    {
        return "KFMNamedScalarDataCollection";
    };


  private:
    std::string fCollectionName;
    std::vector<KFMNamedScalarData> fData;
};


DefineKSAClassName(KFMNamedScalarDataCollection)

}  // namespace KEMField


#endif /* KFMNamedScalarDataCollection_H__ */

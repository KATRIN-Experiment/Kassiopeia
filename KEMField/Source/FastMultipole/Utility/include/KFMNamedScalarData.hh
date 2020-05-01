#ifndef KFMNamedScalarData_HH__
#define KFMNamedScalarData_HH__


#include "KSAStructuredASCIIHeaders.hh"

#include <string>
#include <vector>

namespace KEMField
{

/*
*
*@file KFMNamedScalarData.hh
*@class KFMNamedScalarData
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Apr 23 10:17:37 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMNamedScalarData : public KSAInputOutputObject
{
  public:
    KFMNamedScalarData() : fName(""){};

    KFMNamedScalarData(const KFMNamedScalarData& rhs) : KSAInputOutputObject()
    {
        fName = rhs.fName;
        fData = rhs.fData;
    }

    ~KFMNamedScalarData() override{};

    unsigned int GetSize() const
    {
        return fData.size();
    };

    std::string GetName() const
    {
        return fName;
    };
    void SetName(const std::string& name)
    {
        fName = name;
    };

    void AddNextValue(double value)
    {
        fData.push_back(value);
    };
    unsigned int GetNValues() const
    {
        return fData.size();
    };

    double GetValue(unsigned int i) const
    {
        return fData[i];
    };
    void SetValue(unsigned int i, double data)
    {
        fData[i] = data;
    };

    void GetData(std::vector<double>* data) const
    {
        *data = fData;
    };
    void SetData(const std::vector<double>* data)
    {
        fData = *data;
    };

    void DefineOutputNode(KSAOutputNode* node) const override;
    void DefineInputNode(KSAInputNode* node) override;
    virtual const char* ClassName() const
    {
        return "KFMNamedScalarData";
    };

  private:
    std::string fName;
    std::vector<double> fData;
};


DefineKSAClassName(KFMNamedScalarData)


}  // namespace KEMField

#endif /* KFMNamedScalarData_H__ */

#ifndef KSATestC_HH__
#define KSATestC_HH__

#include "KSAStructuredASCIIHeaders.hh"
#include "KSATestA.hh"

#include <iostream>
#include <utility>
#include <vector>

namespace KEMField
{

/**
*
*@file KSATestC.hh
*@class KSATestC
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 18 09:57:48 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSATestC : public KSATestA
{
  public:
    KSATestC() :
        KSATestA(){

        };


    KSATestC(const KSATestC& copyObject) : KSATestA(copyObject)
    {
        fCData = copyObject.fCData;
    }

    ~KSATestC() override{

    };

    const char* GetName() const override;

    KSATestC& operator=(const KSATestC& rhs)
    {
        if (&rhs != this) {
            KSATestA::operator=(rhs);
            fCData = rhs.fCData;
        }
        return *this;
    }

    double GetCData() const;
    void SetCData(const double& x);

    void DefineOutputNode(KSAOutputNode* node) const override;

    void DefineInputNode(KSAInputNode* node) override;

    const char* ClassName() const override
    {
        return "KSATestC";
    };

  protected:
    double fCData;
};

DefineKSAClassName(KSATestC)


}  // namespace KEMField


#endif /* KSATestC_H__ */

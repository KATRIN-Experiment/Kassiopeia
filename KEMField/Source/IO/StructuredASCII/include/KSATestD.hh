#ifndef KSATestD_HH__
#define KSATestD_HH__


#include "KSAStructuredASCIIHeaders.hh"
#include "KSATestB.hh"

namespace KEMField
{


/**
*
*@file KSATestD.hh
*@class KSATestD
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 16 12:42:15 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSATestD : public KSATestB
{
  public:
    KSATestD()
    {
        ;
    }
    ~KSATestD() override
    {
        ;
    }

    KSATestD(const KSATestD& copyObject) : KSATestB(copyObject)
    {
        fD = copyObject.fD;
    }

    KSATestD& operator=(const KSATestD& rhs)
    {
        if (&rhs != this) {
            fX = rhs.fX;
            fY = rhs.fY;
            fArr[0] = rhs.fArr[0];
            fArr[1] = rhs.fArr[1];
            fArr[2] = rhs.fArr[2];
            fD = rhs.fD;
        }
        return *this;
    }

    void DefineOutputNode(KSAOutputNode* node) const override;

    void DefineInputNode(KSAInputNode* node) override;

    void SetD(const double& d)
    {
        fD = d;
    };
    double GetD() const
    {
        return fD;
    };

    const char* ClassName() const override
    {
        return "KSATestD";
    };

  protected:
    double fD;
};


DefineKSAClassName(KSATestD)

}  // namespace KEMField


#endif /* KSATestD_H__ */

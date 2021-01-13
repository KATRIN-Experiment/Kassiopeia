#ifndef KSATestB_HH__
#define KSATestB_HH__

#include "KSAStructuredASCIIHeaders.hh"

#include <utility>
#include <vector>

namespace KEMField
{


/**
*
*@file KSATestB.hh
*@class KSATestB
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 18 10:00:46 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

//class KSATestB: public KSAFixedSizeInputOutputObject
class KSATestB : public KSAInputOutputObject
{
  public:
    KSATestB()
    {
        fX = 0;
        fY = 0;
    }

    KSATestB(const KSATestB& copyObject) : KSAInputOutputObject()
    {
        fX = copyObject.fX;
        fY = copyObject.fY;
        fArr[0] = copyObject.fArr[0];
        fArr[1] = copyObject.fArr[1];
        fArr[2] = copyObject.fArr[2];
    }

    KSATestB& operator=(const KSATestB& rhs)
    {
        if (&rhs != this) {
            fX = rhs.fX;
            fY = rhs.fY;
            fArr[0] = rhs.fArr[0];
            fArr[1] = rhs.fArr[1];
            fArr[2] = rhs.fArr[2];
        }
        return *this;
    }

    ~KSATestB() override = default;
    ;

    virtual const char* GetName() const;


    double GetX() const;
    void SetX(const double& x);

    double GetY() const;
    void SetY(const double& y);

    void SetArray(const double* arr)
    {
        fArr[0] = arr[0];
        fArr[1] = arr[1];
        fArr[2] = arr[2];
    };

    void GetArray(double* arr) const
    {
        arr[0] = fArr[0];
        arr[1] = fArr[1];
        arr[2] = fArr[2];
    };

    void DefineOutputNode(KSAOutputNode* node) const override;

    void DefineInputNode(KSAInputNode* node) override;

    virtual const char* ClassName() const
    {
        return "KSATestB";
    };

  protected:
    double fX;
    double fY;
    double fArr[3];
};

DefineKSAClassName(KSATestB)


}  // namespace KEMField

#endif /* KSATestB_H__ */

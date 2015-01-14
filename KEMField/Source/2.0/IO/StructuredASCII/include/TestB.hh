#ifndef TestB_HH__
#define TestB_HH__

#include <utility>
#include <vector>

#include "KSAStructuredASCIIHeaders.hh"

//#include "KSAOutputObject.hh"
//#include "KSAInputOutputObject.hh"
//#include "KSAFixedSizeInputOutputObject.hh"

//#include "KSAAssociatedValuePODOutputNode.hh"
//#include "KSAAssociatedReferencePODInputNode.hh"

namespace KEMField{


/**
*
*@file TestB.hh
*@class TestB
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 18 10:00:46 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

//class TestB: public KSAFixedSizeInputOutputObject
class TestB: public KSAInputOutputObject
{
    public:

        TestB()
        {
            fX = 0;
            fY = 0;
        }

        TestB(const TestB& copyObject)
        {
            fX = copyObject.fX;
            fY = copyObject.fY;
            fArr[0] = copyObject.fArr[0];
            fArr[1] = copyObject.fArr[1];
            fArr[2] = copyObject.fArr[2];
        }

        TestB& operator=(const TestB& rhs)
        {
            if(&rhs != this)
            {
                fX = rhs.fX;
                fY = rhs.fY;
                fArr[0] = rhs.fArr[0];
                fArr[1] = rhs.fArr[1];
                fArr[2] = rhs.fArr[2];
            }
            return *this;
        }

        virtual ~TestB()
        {

        };

        virtual const char* GetName() const;


        double GetX() const;
        void SetX(const double& x);

        double GetY() const;
        void SetY(const double& y);

        void SetArray(const double* arr){fArr[0] = arr[0]; fArr[1] = arr[1]; fArr[2] = arr[2];};

        void GetArray(double* arr) const {arr[0] = fArr[0]; arr[1] = fArr[1]; arr[2] = fArr[2];};

        void DefineOutputNode(KSAOutputNode* node) const;

        void DefineInputNode(KSAInputNode* node);

        virtual const char* ClassName() const { return "TestB"; };

    protected:

        double fX;
        double fY;
        double fArr[3];

};

DefineKSAClassName( TestB );


}

#endif /* TestB_H__ */

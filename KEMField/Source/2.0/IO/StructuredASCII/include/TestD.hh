#ifndef TestD_HH__
#define TestD_HH__


#include "TestB.hh"

#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField{


/**
*
*@file TestD.hh
*@class TestD
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 16 12:42:15 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class TestD: public TestB
{
    public:
        TestD(){;}
        virtual ~TestD(){;}

        TestD(const TestD& copyObject):
        TestB(copyObject)
        {
            fD = copyObject.fD;
        }

        TestD& operator=(const TestD& rhs)
        {
            if(&rhs != this)
            {
                fX = rhs.fX;
                fY = rhs.fY;
                fArr[0] = rhs.fArr[0];
                fArr[1] = rhs.fArr[1];
                fArr[2] = rhs.fArr[2];
                fD = rhs.fD;
            }
            return *this;
        }

        virtual void DefineOutputNode(KSAOutputNode* node) const;

        virtual void DefineInputNode(KSAInputNode* node);

        void SetD(const double& d){fD = d;};
        double GetD() const {return fD;};

        virtual const char* ClassName() const { return "TestD"; };

    protected:

        double fD;

};


DefineKSAClassName( TestD );

}



#endif /* TestD_H__ */

#ifndef TestC_HH__
#define TestC_HH__

#include <vector>
#include <utility>
#include <iostream>

#include "TestA.hh"
#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField{

/**
*
*@file TestC.hh
*@class TestC
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 18 09:57:48 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class TestC: public TestA
{
    public:

        TestC():TestA()
        {

        };


        TestC(const TestC& copyObject):
        TestA(copyObject)
        {
            fCData = copyObject.fCData;
        }

        virtual ~TestC()
        {

        };

        virtual const char* GetName() const;

        TestC& operator=(const TestC& rhs)
        {
            if(&rhs != this)
            {
                TestA::operator = (rhs);
                fCData = rhs.fCData;
            }
            return *this;
        }

        double GetCData() const;
        void SetCData(const double& x);

        void DefineOutputNode(KSAOutputNode* node) const;

        void DefineInputNode(KSAInputNode* node);

        virtual const char* ClassName() const { return "TestC"; };

    protected:


        double fCData;

};

DefineKSAClassName( TestC );


}


#endif /* TestC_H__ */

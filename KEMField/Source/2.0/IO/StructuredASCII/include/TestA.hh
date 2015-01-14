#ifndef TestA_HH__
#define TestA_HH__

#include <vector>
#include <utility>
#include <iostream>

#include "TestB.hh"

#include "KSAStructuredASCIIHeaders.hh"

//#include "KSAOutputObject.hh"
//#include "KSAInputObject.hh"
//#include "KSAAssociatedPointerObjectOutputNode.hh"
//#include "KSAAssociatedPointerPODOutputNode.hh"

//#include "KSAAssociatedReferencePODInputNode.hh"
//#include "KSAAssociatedPointerPODInputNode.hh"
//#include "KSAObjectInputNode.hh"


namespace KEMField{

/**
*
*@file TestA.hh
*@class TestA
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 18 09:57:48 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class TestA: public KSAOutputObject, public KSAInputObject
{
    public:

        TestA():KSAOutputObject()
        {

        };


        TestA(const TestA& copyObject):
        fB(copyObject.fB)
        {
            fData = copyObject.fData;
            fBVec = copyObject.fBVec;
        }

        virtual ~TestA()
        {

        };

        virtual const char* GetName() const;

        void AddData(double data);
        void ClearData();

//        void GetData(std::vector<double>* data) const ;
        const std::vector<double>* GetData() const;
        void SetData(const std::vector<double>* data);

        const TestB* GetB() const;
        void SetB(const TestB& b);

        void ClearBVector()
        {
            fBVec.clear(); //yeah this is a memory leak, but i am too lazy to fix this for a test example
        }

        void AddBVector(std::vector< TestB* >* vec)
        {
            fBVec.push_back(*vec);
        }


        TestA& operator=(const TestA& rhs)
        {
            if(&rhs != this)
            {
                fData = rhs.fData;
                fB = rhs.fB;
                fBVec = rhs.fBVec;
            }
            return *this;
        }

        virtual void DefineOutputNode(KSAOutputNode* node) const;

        virtual void DefineInputNode(KSAInputNode* node);


        virtual const char* ClassName() const { return "TestA"; };

    protected:

        TestB fB;
        std::vector<double> fData;
        std::vector< std::vector< TestB* > > fBVec;

};

DefineKSAClassName( TestA );


}


#endif /* TestA_H__ */

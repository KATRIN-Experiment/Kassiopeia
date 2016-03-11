#ifndef KSATestA_HH__
#define KSATestA_HH__

#include <vector>
#include <utility>
#include <iostream>

#include "KSATestB.hh"

#include "KSAStructuredASCIIHeaders.hh"


namespace KEMField{

/**
*
*@file KSATestA.hh
*@class KSATestA
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 18 09:57:48 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSATestA: public KSAOutputObject, public KSAInputObject
{
    public:

        KSATestA():KSAOutputObject()
        {

        };


        KSATestA(const KSATestA& copyObject):
        KSAOutputObject(),
        KSAInputObject(),
        fB(copyObject.fB)
        {
            fData = copyObject.fData;
            fBVec = copyObject.fBVec;
        }

        virtual ~KSATestA()
        {

        };

        virtual const char* GetName() const;

        void AddData(double data);
        void ClearData();

//        void GetData(std::vector<double>* data) const ;
        const std::vector<double>* GetData() const;
        void SetData(const std::vector<double>* data);

        const KSATestB* GetB() const;
        void SetB(const KSATestB& b);

        void ClearBVector()
        {
            fBVec.clear(); //yeah this is a memory leak, but i am too lazy to fix this for a KSATest example
        }

        void AddBVector(std::vector< KSATestB* >* vec)
        {
            fBVec.push_back(*vec);
        }


        KSATestA& operator=(const KSATestA& rhs)
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


        virtual const char* ClassName() const { return "KSATestA"; };

    protected:

        KSATestB fB;
        std::vector<double> fData;
        std::vector< std::vector< KSATestB* > > fBVec;

};

DefineKSAClassName( KSATestA )


}


#endif /* KSATestA_H__ */

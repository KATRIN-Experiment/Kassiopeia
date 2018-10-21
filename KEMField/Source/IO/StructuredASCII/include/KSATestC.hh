#ifndef KSATestC_HH__
#define KSATestC_HH__

#include <vector>
#include <utility>
#include <iostream>

#include "KSATestA.hh"
#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField{

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

class KSATestC: public KSATestA
{
    public:

        KSATestC():KSATestA()
        {

        };


        KSATestC(const KSATestC& copyObject):
        KSATestA(copyObject)
        {
            fCData = copyObject.fCData;
        }

        virtual ~KSATestC()
        {

        };

        virtual const char* GetName() const;

        KSATestC& operator=(const KSATestC& rhs)
        {
            if(&rhs != this)
            {
                KSATestA::operator = (rhs);
                fCData = rhs.fCData;
            }
            return *this;
        }

        double GetCData() const;
        void SetCData(const double& x);

        void DefineOutputNode(KSAOutputNode* node) const;

        void DefineInputNode(KSAInputNode* node);

        virtual const char* ClassName() const { return "KSATestC"; };

    protected:


        double fCData;

};

DefineKSAClassName( KSATestC )


}


#endif /* KSATestC_H__ */

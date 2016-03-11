#ifndef KSATestD_HH__
#define KSATestD_HH__


#include "KSATestB.hh"

#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField{


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

class KSATestD: public KSATestB
{
    public:
        KSATestD(){;}
        virtual ~KSATestD(){;}

        KSATestD(const KSATestD& copyObject):
        KSATestB(copyObject)
        {
            fD = copyObject.fD;
        }

        KSATestD& operator=(const KSATestD& rhs)
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

        virtual const char* ClassName() const { return "KSATestD"; };

    protected:

        double fD;

};


DefineKSAClassName( KSATestD )

}



#endif /* KSATestD_H__ */

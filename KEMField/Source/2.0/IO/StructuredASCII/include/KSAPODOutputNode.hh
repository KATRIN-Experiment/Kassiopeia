#ifndef KSAPODOutputNode_HH__
#define KSAPODOutputNode_HH__

#include "KSAPODConverter.hh"
#include "KSAOutputNode.hh"

namespace KEMField{


/**
*
*@file KSAPODOutputNode.hh
*@class KSAPODOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Dec 28 23:28:49 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


//should work as long as U is a pod type of a std::vector<> of a pod type
template < typename U >
class KSAPODOutputNode: public KSAOutputNode
{
    public:


        KSAPODOutputNode(std::string name):KSAOutputNode(name)
        {
            fConverter = new KSAPODConverter< U >();
            fStringValue = "INVALID";
        };

        virtual ~KSAPODOutputNode()
        {
            delete fConverter;
        }

        virtual bool TagsAreSuppressed(){return false;};

        virtual bool IsComposite(){return false;};

        void SetValue(const U& val)
        {
            fConverter->ConvertParameterToString(fStringValue, val);
        }

        void SetValue(const U* val)
        {
            fConverter->ConvertParameterToString(fStringValue, val);
        }

    protected:

        virtual std::string GetSingleLine(){return fStringValue;};

        std::string fStringValue;
        KSAPODConverter< U >* fConverter;

    private:

        //cannot instantiate without providing a name
        KSAPODOutputNode(){;};

};


}//end of kemfield namespace


#endif /* KSAPODOutputNode_H__ */

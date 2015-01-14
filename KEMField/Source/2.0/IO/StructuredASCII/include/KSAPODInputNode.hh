#ifndef KSAPODInputNode_HH__
#define KSAPODInputNode_HH__

#include "KSAPODConverter.hh"
#include "KSAInputNode.hh"

namespace KEMField{


/**
*
*@file KSAPODInputNode.hh
*@class KSAPODInputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Dec 28 23:28:49 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


//should work as long as U is a pod type or a std::vector<> of a pod type
template < typename U >
class KSAPODInputNode: public KSAInputNode
{
    public:


        KSAPODInputNode(std::string name):KSAInputNode(name)
        {
            fConverter = new KSAPODConverter< U >();
            fChildren.clear();
        };

        virtual ~KSAPODInputNode()
        {
            delete fConverter;
        }

        virtual bool TagsAreSuppressed(){return true;};

        virtual bool IsComposite(){return false;};

        virtual void AddChild(KSAInputNode* /*child*/)
        {
            //no children allowed
        };

        //next node will be set to NULL if the visitor traversing the tree
        //needs to move back to the parent, or stay on the current node
        virtual int GetNextNode(KSAInputNode*& next_node)
        {
            next_node = fNextNode;
            return fStatus;
        }

        virtual bool HasChildren() const
        {
            return false;
        };

        void FinalizeObject(){;};

        void AddLine(const std::string& line)
        {
            //assumes the line has been trimmed of uncessary whitespace!!
            if( LineIsStopTag(line) )
            {
                fStatus = KSANODE_MOVE_UPWARD;
                fNextNode = NULL;
                FinalizeObject();
            }
            else
            {
                //must be the string equivalent of the parameter
                fConverter->ConvertStringToParameter(line, fValue);
                fStatus = KSANODE_STAY; //wait for stop tag
                fNextNode = NULL;
            }
        }

    protected:

        U fValue;
        KSAPODConverter< U >* fConverter;

    private:

        //cannot instantiate without providing a name
        KSAPODInputNode(){;};

};


}//end of kemfield namespace


#endif /* KSAPODInputNode_H__ */

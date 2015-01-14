#ifndef KSAOutputNode_HH__
#define KSAOutputNode_HH__

#include <vector>

#include "KSAObject.hh"
#include <vector>

#define KSANODE_MOVE_DOWNWARD -1 //indicates we need to decend to a child node
#define KSANODE_MOVE_UPWARD 1 //indicates we need to ascend to parent
#define KSANODE_STAY 0 //indicates we need to stay on the current node

namespace KEMField{


/**
*
*@file KSAOutputNode.hh
*@class KSAOutputNode
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Dec 27 23:05:43 EST 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAOutputNode: public KSAObject
{
    public:

        KSAOutputNode(std::string name):
        KSAObject(name),
        fStatus(KSANODE_STAY),
        fSingle(0),
        fIndex(0),
        fNextNode(NULL)
        {
            fChildren.clear();
        };

        KSAOutputNode():
        KSAObject(),
        fStatus(KSANODE_STAY),
        fSingle(0),
        fIndex(0),
        fNextNode(NULL)
        {
            fChildren.clear();
        };

        virtual ~KSAOutputNode()
        {
            for(unsigned int i = 0; i<fChildren.size(); i++)
            {
                delete fChildren[i];
		        fChildren[i] = NULL;
            }

            fChildren.clear();
        };

        virtual void Initialize(){;};

        virtual bool TagsAreSuppressed(){return false;};

        virtual bool IsComposite(){return false;};

        virtual void AddChild(KSAOutputNode* child){fChildren.push_back(child);};

        //next node will be set to NULL if the visitor traversing the tree
        //needs to move back to the parent, or stay on the current node
        virtual int GetNextNode(KSAOutputNode*& next_node)
        {
            next_node = fNextNode;
            return fStatus;
        }

        virtual void Reset()
        {
            fIndex = 0;
            fStatus = KSANODE_STAY;
            fSingle = 0;
            fNextNode = NULL;
            for(unsigned int i = 0; i<fChildren.size(); i++)
            {
                delete fChildren[i];
            }

            fChildren.clear();
        }

        virtual bool HasChildren() const
        {
            if( fChildren.size() == 0 ){return false;};
            return true;
        };

        virtual void GetLine(std::string& line)
        {
            if(fChildren.size() != 0)
            {
                //this iterates over all the children of a composite node
 	        if( fIndex >=0 && fIndex < ((int)fChildren.size()) )
                {
                    //open the next node
                    line = fChildren[fIndex]->GetStartTag() + std::string(LINE_DELIM);
                    fStatus = KSANODE_MOVE_DOWNWARD;
                    fNextNode = fChildren[fIndex];
                    fIndex++;
                }
                else
                {
                    line = GetStopTag() + std::string(LINE_DELIM); //close out the current node
                    fStatus = KSANODE_MOVE_UPWARD;
                    fNextNode = NULL; //next node is parent
                }
            }
            else
            {
                //this is here specifically for POD and non-composite types that
                //can be stringified into a single line without child nodes
                if(fSingle == 0)
                {
                    line = GetSingleLine() + std::string(LINE_DELIM);
                    fStatus = KSANODE_STAY;
                    fNextNode = NULL;
                    fSingle = 1;
                }
                else
                {
                    line = GetStopTag() + std::string(LINE_DELIM);
                    fStatus = KSANODE_MOVE_UPWARD;
                    fNextNode = NULL; //next node is parent
                }

            }
        }


    protected:

        //must defined in a POD node, in composite nodes it is never called
        virtual std::string GetSingleLine(){return std::string("INVALID");};

        int fStatus;
        int fSingle;
        int fIndex;
        KSAOutputNode* fNextNode;
        std::vector< KSAOutputNode* > fChildren;
};


}//end of kemfield namespace


#endif /* KSAOutputNode_H__ */

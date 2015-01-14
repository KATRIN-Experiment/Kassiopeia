#include "KSAInputCollector.hh"
#include <iostream>

namespace KEMField{

KSAInputCollector::KSAInputCollector()
{
    fReader = NULL;
}

KSAInputCollector::~KSAInputCollector()
{

}

void
KSAInputCollector::ForwardInput(KSAInputNode* root)
{

    fNodeStack = std::stack< KSAInputNode* >();
    fNodeStack.push(root);

    bool file_status = false;
    if(fReader != NULL){file_status = true;};


    int count = 0;
    while(fNodeStack.size() != 0 && file_status)
    {

        //std::cout<<"calling the reader at line count: "<<count<<std::endl;

        file_status = fReader->GetLine(fLine);


        fNodeStack.top()->AddLine(fLine);

        fTempNode = NULL;
        //now figure out whether we needed to decend the tree, stay, or ascend
        fStatus = fNodeStack.top()->GetNextNode(fTempNode);

        if(fStatus == KSANODE_MOVE_DOWNWARD && fTempNode != NULL)
        {
            //std::cout<<"moving downard to node: "<<fTempNode->GetName()<<std::endl;
            fNodeStack.push( fTempNode ); //descend to child
        }
        else if (fStatus == KSANODE_MOVE_UPWARD)
        {
            //std::cout<<"moving upward to node: "<<fNodeStack.top()->GetName()<<std::endl;
            fNodeStack.pop();
        }
        else if (fStatus == KSANODE_STAY)
        {
            //do nothing, stay on same node for another line
        }
        else
        {
            //std::cout<<"Error!"<<std::endl;
            break;
            //break, error
        }

        //std::cout<<"line = "<<fLine<<std::endl;

        count++;

    };

}

}//end of namespace

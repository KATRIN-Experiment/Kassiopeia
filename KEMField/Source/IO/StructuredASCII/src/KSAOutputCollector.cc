#include "KSAOutputCollector.hh"

namespace KEMField
{

void KSAOutputCollector::SetFileWriter(KSAFileWriter* writer)
{
    fWriter = writer;
}

//recursively apply the operational visitor to every node
//below this one
void KSAOutputCollector::CollectOutput(KSAOutputNode* node)
{
    fNodeStack = std::stack<KSAOutputNode*>();
    fNodeStack.push(node);

    //add the root's start tag
    fLine = node->GetStartTag() + std::string(LINE_DELIM);

    fWriter->AddToFile(fLine);

    do {
        CollectNodeOutput(fNodeStack.top());
        fTempNode = nullptr;
        //now figure out whether we needed to decend the tree, stay, or ascend
        fStatus = fNodeStack.top()->GetNextNode(fTempNode);

        ForwardNodeOutput();

        if (fStatus == KSANODE_MOVE_DOWNWARD && fTempNode != nullptr) {
            fNodeStack.push(fTempNode);  //descend to child
        }
        else if (fStatus == KSANODE_MOVE_UPWARD) {
            fNodeStack.pop();
        }
        else if (fStatus == KSANODE_STAY) {
            //do nothing, stay on same node for another line
        }
        else {
            //break, error
        }
    } while (!fNodeStack.empty());
}

void KSAOutputCollector::CollectNodeOutput(KSAOutputNode* node)
{
    node->GetLine(fLine);
}


void KSAOutputCollector::ForwardNodeOutput()
{
    fStream.str("");
    fStream.clear();


    //prepend the line with the same number of tabs,
    //as the current depth of the output tree
    //this has no use other than to make the output look pretty

    if (fUseTabbing) {
        int depth = fNodeStack.size();
        if (fStatus == KSANODE_MOVE_UPWARD) {
            if (depth > 0) {
                depth -= 1;
            }
        }
        for (int i = 0; i < depth; i++) {
            fStream << "\t";
        }
    }

    fStream << fLine;
    fWriter->AddToFile(fStream.str());
}


}  // namespace KEMField

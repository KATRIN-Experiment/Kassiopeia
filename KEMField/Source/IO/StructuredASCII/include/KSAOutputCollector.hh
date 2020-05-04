#ifndef KSAOutputCollector_H__
#define KSAOutputCollector_H__

#include "KSAFileWriter.hh"
#include "KSAOutputNode.hh"

#include <sstream>
#include <stack>
#include <string>

namespace KEMField
{

/**
*
*@file KSAOutputCollector.hh
*@class KSAOutputCollector
*@brief visits each node of an output tree below the
* given node in a depth first manner and pipes the output to the file writer
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 28 15:47:04 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAOutputCollector
{
  public:
    KSAOutputCollector()
    {
        fUseTabbing = false;
    };
    ~KSAOutputCollector()
    {
        ;
    };

    void SetFileWriter(KSAFileWriter* writer);

    //recursively ask GetLine() from every node
    //below this one, this should only be called on the root node
    void CollectOutput(KSAOutputNode* node);

    void SetUseTabbingTrue()
    {
        fUseTabbing = true;
    }
    void SetUseTabbingFalse()
    {
        fUseTabbing = false;
    }


  protected:
    void CollectNodeOutput(KSAOutputNode* node);
    void ForwardNodeOutput();

    std::stack<KSAOutputNode*> fNodeStack;
    KSAOutputNode* fTempNode;
    bool fUseTabbing;
    int fStatus;
    std::string fLine;
    KSAFileWriter* fWriter;
    std::stringstream fStream;
};


}  // namespace KEMField

#endif /* __KSAOutputCollector_H__ */

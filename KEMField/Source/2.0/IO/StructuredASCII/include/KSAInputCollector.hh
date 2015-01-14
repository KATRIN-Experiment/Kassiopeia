#ifndef KSAInputCollector_HH__
#define KSAInputCollector_HH__


#include "KSAFileReader.hh"
#include "KSAInputNode.hh"
#include <stack>

namespace KEMField{


/**
*
*@file KSAInputCollector.hh
*@class KSAInputCollector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jan  4 13:55:45 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KSAInputCollector
{
    public:
        KSAInputCollector();
        virtual ~KSAInputCollector();

        void SetFileReader(KSAFileReader* reader){fReader = reader;};

        void ForwardInput(KSAInputNode* root);

    protected:

        std::stack< KSAInputNode* > fNodeStack;
        KSAInputNode* fTempNode;
        int fStatus;
        std::string fLine;
        KSAFileReader* fReader;

};


}


#endif /* KSAInputCollector_H__ */

#ifndef KFMElectrostaticNodeInspector_H
#define KFMElectrostaticNodeInspector_H


#include <vector>
#include <sstream>
#include <string>

#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"
#include "KFMElectrostaticNode.hh"



namespace KEMField{

/**
*
*@file KFMElectrostaticNodeInspector.hh
*@class KFMElectrostaticNodeInspector
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Aug 23 22:35:12 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticNodeInspector: public KFMNodeActor< KFMElectrostaticNode >
{
    public:
        KFMElectrostaticNodeInspector();
        virtual ~KFMElectrostaticNodeInspector();

        virtual void ApplyAction(KFMElectrostaticNode* node);
        void Print();

    private:

        double fNumberOfNodes;

        std::vector< double > fNumberOfNodesAtLevel;

        std::vector< double > fNumberOfElementsAtLevel;

        std::vector< std::vector< double > > fElementSizeAtLevel;

        std::vector< std::vector< double > > fDirectCallDistribution;



};

}//end of KEMField namespace

#endif /* KFMElectrostaticNodeInspector_H */

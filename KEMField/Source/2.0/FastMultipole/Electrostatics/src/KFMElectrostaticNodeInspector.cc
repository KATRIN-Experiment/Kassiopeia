#include "KFMElectrostaticNodeInspector.hh"

using namespace KEMField;

KFMElectrostaticNodeInspector::KFMElectrostaticNodeInspector()
{
    double fNumberOfNodes;
    fNumberOfNodesAtLevel.clear();
    fNumberOfElementsAtLevel.clear();
    fElementSizeAtLevel.clear();
    fDirectCallDistribution.clear();
}

KFMElectrostaticNodeInspector::~KFMElectrostaticNodeInspector()
{

}


void
KFMElectrostaticNodeInspector::ApplyAction(KFMElectrostaticNode* node);
{
    fNumberOfNodes++;


    if(node != NULL)
    {

        //figure out the number of Elements it owns
        int n_elements = 0;
        KFMIdentitySet* id_set = KFMObjectRetriever<KFMElectrostaticNodeObjects,KFMIdentitySet >::GetNodeObject(node);
        if(id_set != NULL)
        {
            n_elements =  id_set->GetSize();
        }

        //figure out which level it belongs to
        int level = node->GetLevel();

        if(level < fNumberOfNodesAtLevel.size() )
        {
            fNumberOfNodesAtLevel[level] += 1; //add it to the count for that level
            fNumberOfElementsAtLevel[level] += n_elements;
        }
        else
        {
            for(unsigned int i=fNumberOfNodesAtLevel.size(); i <= level; i++)
            {
                //haven't see this level before, so add it
                fNumberOfNodesAtLevel.push_back(0);
                fNumberOfElementsAtLevel.push_back(0);
                fElementSizeAtLevel.push_back( std::vector<double>() );
                fDirectCallDistribution.push_back( std::vector<double>() );
            }

            fNumberOfNodesAtLevel[level] += 1; //add it to the count for that level
            fNumberOfElementsAtLevel[level] += n_elements;
        }

        //compute the number of direct calls this node has to make


        int n_direct_calls = node->GetNodeObject()->GetNumberOfNearbyElements();
        KFMElectrostaticNode* temp = node->GetParent();

        while(temp != NULL)
        {
            n_direct_calls += temp->GetNodeObject()->GetNumberOfNearbyElements();
            temp = temp->GetParent();
        }

        fTotalNumberOfDirectCalls += n_direct_calls;

        if(n_direct_calls > fMaxNumberOfDirectCalls)
        {
            fMaxNumberOfDirectCalls = n_direct_calls;
        }

        if(n_direct_calls < fMinNumberOfDirectCalls)
        {
            fMinNumberOfDirectCalls = n_direct_calls;
        }

        if(n_direct_calls !=0)
        {
            double interval = ( (double)n_direct_calls/(double)fNElements );
            int interval_index = std::floor( 50*interval );
            fDirectCallDist[interval_index] += 1;
            fLevelElementCount[node->GetLevel()] += node->GetNodeObject()->GetNElements();
        }
        else
        {
            fNNodesWithNoDirectCalls++;
        }
    }

    fLevelNodeCount[node->GetLevel()] += 1;

    fLevelNodeLength[node->GetLevel()] = node->GetNodeObject()->GetLength();

    //not that this average size is not quite correct, since it is not updated after Elements are moved down to children
    fLevelAvePrimSize[node->GetLevel()] += (node->GetNodeObject()->GetAverageElementSize())*(node->GetNodeObject()->GetNElements());

}


void
KFMElectrostaticNodeInspector::Print()
{
    std::stringstream msg;

    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), std::string("Region tree constructed with the following properties: \n"), 0, 1);

    msg.str("");
    msg<<"Tree has a total of "<<fNumberOfNodes<<" nodes.";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);

    msg.str("");
    msg<<"Tree has a total of "<<fNumberOfNodesWithElements<<" nodes which contain Elements. \n";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);

    msg.str("");
    msg<<"Estimated response function memory usage is "<<(fResponseMem/(1000.*1000.))<<" MB";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);
    msg.str("");
    msg<<"Estimated region tree memory usage is "<<(fRegionMem/(1000.*1000.))<<" MB";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);
    msg.str("");
    msg<<"Estimated total memory usage by FFTM is "<<(fRegionMem/(1000.*1000.))+(fResponseMem/(1000.*1000.))<<" MB \n";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);


    for(unsigned int i=0; i<= fMaxTreeLevel; i++)
    {
        msg.str("");
        if(i != fMaxTreeLevel)
        {
            double ave_size = fLevelAvePrimSize[i]/((double)fLevelElementCount[i]);
            if(fLevelElementCount[i] == 0)
            {
                ave_size = 0;
            }
            msg<<"Level "<<i<<" has "<<fLevelElementCount[i]<<" Elements in "<<fLevelNodeCount[i]<<" nodes of length "<<fLevelNodeLength[i]<<", with average Element radius of "<<ave_size<<".";
        }
        else
        {
            double ave_size = fLevelAvePrimSize[i]/((double)fLevelElementCount[i]);
            if(fLevelElementCount[i] == 0)
            {
                ave_size = 0;
            }
            msg<<"Level "<<i<<" has "<<fLevelElementCount[i]<<" Elements in "<<fLevelNodeCount[i]<<" nodes of length "<<fLevelNodeLength[i]<<", with average Element radius of "<<ave_size<<". \n";
        }
        KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);
    }


    msg.str("");
    msg<<"Max number of direct calls from a single node is "<<fMaxNumberOfDirectCalls;
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);

    msg.str("");
    msg<<"Min number of direct calls from a single node is "<<fMinNumberOfDirectCalls;
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);

    msg.str("");
    msg<<"Average number of direct calls from a single node is "<<(double)fTotalNumberOfDirectCalls/(double)fNumberOfNodes<<" \n";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);


    double no_call_percent = ((double)fNNodesWithNoDirectCalls/(double)fNumberOfNodes)*100;
    double interval_percents[50];
    for(int i=0; i<50; i++)
    {
        interval_percents[i] = ((double)fDirectCallDist[i]/(double)fNumberOfNodes)*100;
    }


    msg.str("");
    msg<<"Direct Call Distribution:";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);

    msg.str("");
    msg << no_call_percent;
    msg << "% of nodes call 0% of all Elements.";
    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);

    double sum = no_call_percent;
    Bool_t collected_enough = false;
    Bool_t finished = false;
    if(sum >= 99.999)
    {
        collected_enough = true;
    }
    for(int j=0; j < 50; j++)
    {
        if(!finished)
        {
            if(!collected_enough)
            {
                msg.str("");
                msg << interval_percents[j];
                msg << "% of nodes call between "<<2*j<<"% and "<<2*(j+1)<<"% of all Elements.";
                KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);
            }
            else
            {
                msg.str("");
                if( (100 - sum) < 100. - 99.999 )
                {
                    msg << 0.0;
                }
                else
                {
                    msg << (100. - 99.999);
                }

                msg << "% of nodes call more than "<<2*j<<"% of all Elements.";
                KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), msg.str(), 0, 2);
                finished = true;
            }
            sum += interval_percents[j];
            if(sum >= 99.999)
            {
                collected_enough = true;
            }
        }

    }




    KIOManager::GetInstance()->Message(std::string("KFMElectrostaticNodeInspector"), std::string("Print"), std::string(""), 0, 3);

}

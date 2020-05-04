#ifndef KFMCompoundActor_HH__
#define KFMCompoundActor_HH__

#include "KFMNodeActor.hh"

#include <vector>

namespace KEMField
{

/*
*
*@file KFMCompoundActor.hh
*@class KFMCompoundActor
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Aug 12 12:05:27 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename NodeType> class KFMCompoundActor : public KFMNodeActor<NodeType>
{
  public:
    KFMCompoundActor()
    {
        ;
    };
    ~KFMCompoundActor() override
    {
        ;
    };

    //add a visitor to the back of the list of actors
    void AddActor(KFMNodeActor<NodeType>* actor)
    {
        if (actor != this && actor != nullptr)  //avoid a disaster
        {
            fActors.push_back(actor);
        }
    };

    void Clear()
    {
        fActors.clear();
    }

    void ApplyAction(NodeType* node) override
    {
        for (unsigned int i = 0; i < fActors.size(); i++) {
            fActors[i]->ApplyAction(node);
        }
    }

  private:
    std::vector<KFMNodeActor<NodeType>*> fActors;
};

}  // namespace KEMField


#endif /* KFMCompoundActor_H__ */

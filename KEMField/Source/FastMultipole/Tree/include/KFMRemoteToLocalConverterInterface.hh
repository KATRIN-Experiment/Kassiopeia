#ifndef __KFMRemoteToLocalConverterInterface_H__
#define __KFMRemoteToLocalConverterInterface_H__

#include "KFMNodeActor.hh"
#include "KFMObjectRetriever.hh"

namespace KEMField
{

/**
*
*@file KFMRemoteToLocalConverterInterface.hh
*@class KFMRemoteToLocalConverterInterface
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jan 31 12:02:08 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/


template<typename ObjectTypeList, unsigned int SpatialNDIM, typename M2LType>
class KFMRemoteToLocalConverterInterface : public KFMNodeActor<KFMNode<ObjectTypeList>>
{
  public:
    KFMRemoteToLocalConverterInterface()
    {
        fTopLevelM2LConverter = new M2LType();
        fM2LConverter = new M2LType();
    }

    ~KFMRemoteToLocalConverterInterface() override
    {
        delete fTopLevelM2LConverter;
        delete fM2LConverter;
    }

    //direct access to the M2L converters
    M2LType* GetTopLevelM2LConverter()
    {
        return fTopLevelM2LConverter;
    };
    M2LType* GetTreeM2LConverter()
    {
        return fM2LConverter;
    };

    bool IsScaleInvariant() const
    {
        return fM2LConverter->IsScaleInvariant();
    };

    //set the world volume length
    void SetLength(double length)
    {
        fTopLevelM2LConverter->SetLength(length);
        fM2LConverter->SetLength(length);
    };

    //set the maximum depth of the tree
    void SetMaxTreeDepth(unsigned int max_depth)
    {
        fTopLevelM2LConverter->SetMaxTreeDepth(1);
        fM2LConverter->SetMaxTreeDepth(max_depth);
    };

    virtual bool IsFinished() const
    {
        return (fTopLevelM2LConverter->IsFinished() && fM2LConverter->IsFinished());
    };

    virtual void Prepare()
    {
        fTopLevelM2LConverter->Prepare();
        fM2LConverter->Prepare();
    };

    virtual void Finalize()
    {
        fTopLevelM2LConverter->Finalize();
        fM2LConverter->Finalize();
    };

    ////////////////////////////////////////////////////////////////////////
    void SetNumberOfTermsInSeries(unsigned int n_terms)
    {
        fTopLevelM2LConverter->SetNumberOfTermsInSeries(n_terms);
        fM2LConverter->SetNumberOfTermsInSeries(n_terms);
    };

    void SetZeroMaskSize(int zeromasksize)
    {
        fTopLevelM2LConverter->SetZeroMaskSize(zeromasksize);
        fTopLevelM2LConverter->SetNeighborOrder(0);

        fM2LConverter->SetZeroMaskSize(zeromasksize);
        fM2LConverter->SetNeighborOrder(zeromasksize);
    }

    void SetDivisions(int div)
    {
        fM2LConverter->SetDivisions(div);
    }

    void SetTopLevelDivisions(int div)
    {
        fTopLevelM2LConverter->SetDivisions(div);
    }


    virtual void Initialize()
    {
        fTopLevelM2LConverter->Initialize();
        fM2LConverter->Initialize();
    }

    void ApplyAction(KFMNode<ObjectTypeList>* node) override
    {
        if (node != nullptr) {
            if (node->GetLevel() == 0) {
                //apply the top level action to the root node
                fTopLevelM2LConverter->ApplyAction(node);
            }
            else {
                //apply normal action to all other nodes
                fM2LConverter->ApplyAction(node);
            }
        }
    }

  protected:
    M2LType* fTopLevelM2LConverter;
    M2LType* fM2LConverter;
};

}  // namespace KEMField


#endif /* __KFMRemoteToLocalConverterInterface_H__ */

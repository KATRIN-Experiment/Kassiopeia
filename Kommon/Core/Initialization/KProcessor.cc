#include "KProcessor.hh"

#include <cstdlib>

namespace katrin
{

    KProcessor::KProcessor() :
        fParent( NULL ),
        fChild( NULL )
    {
    }
    KProcessor::~KProcessor()
    {
    }

    void KProcessor::Connect( KProcessor* aParent, KProcessor* aChild )
    {
        if( (aParent == NULL) || (aChild == NULL) )
        {
            return;
        }

        if( (aParent->fChild != NULL) || (aChild->fParent != NULL) )
        {
            return;
        }

        aParent->fChild = aChild;
        aChild->fParent = aParent;

        return;
    }
    void KProcessor::Disconnect( KProcessor* aParent, KProcessor* aChild )
    {
        if( (aParent == NULL) || (aChild == NULL) )
        {
            return;
        }

        if( (aParent->fChild != aChild) || (aChild->fParent != aParent) )
        {
            return;
        }

        aParent->fChild = NULL;
        aChild->fParent = NULL;

        return;
    }

    void KProcessor::InsertBefore( KProcessor* aTarget )
    {
        if( (fParent != NULL) || (fChild != NULL) || (aTarget == NULL) )
        {
            return;
        }

        if( aTarget->fParent != NULL )
        {
            fParent = aTarget->fParent;
            fParent->fChild = this;
        }

        fChild = aTarget;
        aTarget->fParent = this;

        return;
    }
    void KProcessor::InsertAfter( KProcessor* aTarget )
    {
        if( (fParent != NULL) || (fChild != NULL) || (aTarget == NULL) )
        {
            return;
        }

        if( aTarget->fChild != NULL )
        {
            fChild = aTarget->fChild;
            fChild->fParent = this;
        }

        fParent = aTarget;
        aTarget->fChild = this;

        return;
    }
    void KProcessor::Remove()
    {
        if( (fParent != NULL) && (fChild != NULL) )
        {
            fParent->fChild = fChild;
            fChild->fParent = fParent;

            fParent = NULL;
            fChild = NULL;

            return;
        }

        if( fParent != NULL )
        {
            fParent->fChild = NULL;
            fParent = NULL;
        }

        if( fChild != NULL )
        {
            fChild->fParent = NULL;
            fChild = NULL;
        }

        return;
    }

    KProcessor* KProcessor::GetFirstParent()
    {
        if( fParent != NULL )
        {
            return fParent->GetFirstParent();
        }
        return this;
    }
    KProcessor* KProcessor::GetParent()
    {
        return fParent;
    }

    KProcessor* KProcessor::GetLastChild()
    {
        if( fChild != NULL )
        {
            return fChild->GetLastChild();
        }
        return this;
    }
    KProcessor* KProcessor::GetChild()
    {
        return fChild;
    }

    void KProcessor::ProcessToken( KBeginParsingToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KBeginFileToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KMidElementToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KElementDataToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KEndElementToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KEndFileToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KEndParsingToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KCommentToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }
    void KProcessor::ProcessToken( KErrorToken* aToken )
    {
        if( fChild == NULL )
        {
            return;
        }
        else
        {
            fChild->ProcessToken( aToken );
            return;
        }
    }

}

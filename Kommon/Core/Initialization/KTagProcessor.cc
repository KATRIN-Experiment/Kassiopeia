#include "KTagProcessor.hh"
#include "KInitializationMessage.hh"

#include "KTagged.h"
using katrin::KTagged;

#include <cstdlib>

namespace katrin
{

    KTagProcessor::KTagProcessor() :
        KProcessor(),
        fState( eInactive ),
        fTags(),
        fTagIt(),
        fTagStack()
    {
    }
    KTagProcessor::~KTagProcessor()
    {
    }

    void KTagProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        if( fState == eInactive )
        {
            if( aToken->GetValue() == "tag" )
            {
                initmsg_debug( "tagging system active" << ret );
                initmsg_debug( "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom );
                fState = eActive;
                return;
            }

            KProcessor::ProcessToken( aToken );
            return;
        }

        return;
    }
    void KTagProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
        if( fState == eActive )
        {
            if( aToken->GetValue() != "name" )
            {
                initmsg( eError ) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
                initmsg( eError ) << "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
                return;
            }
            initmsg_debug( "tagging system starting tag" << ret );
            initmsg_debug( "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom );
            return;
        }

        KProcessor::ProcessToken( aToken );
        return;
    }
    void KTagProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        if( fState == eActive )
        {
            initmsg_debug( "tagging system added tag <" << aToken->GetValue() << ">" << ret );
            initmsg_debug( "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom );
            fTags.push_back( aToken->GetValue() );
            return;
        }

        KProcessor::ProcessToken( aToken );
        return;
    }
    void KTagProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
        if( fState == eActive )
        {
            initmsg_debug( "tagging system ending tag" << ret );
            initmsg_debug( "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom );
            return;
        }

        KProcessor::ProcessToken( aToken );
        return;
    }
    void KTagProcessor::ProcessToken( KMidElementToken* aToken )
    {
        if( fState == eActive )
        {
            initmsg_debug( "tagging system opening tags <" );
            for( fTagIt = fTags.begin(); fTagIt != fTags.end(); fTagIt++ )
            {
                initmsg_debug( " " << *fTagIt );
                KTagged::OpenTag( *fTagIt );
            }
            initmsg_debug( " >" << ret );
            initmsg_debug( "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom );
            fTagStack.push( fTags );
            fTags.clear();
            fState = eInactive;
            return;
        }

        KProcessor::ProcessToken( aToken );
        return;
    }
    void KTagProcessor::ProcessToken( KEndElementToken* aToken )
    {
        if( fState == eInactive )
        {
            if( aToken->GetValue() == "tag" )
            {
                initmsg_debug( "tagging system closing tags <" );
                fTags = fTagStack.top();
                for( fTagIt = fTags.begin(); fTagIt != fTags.end(); fTagIt++ )
                {
                    initmsg_debug( " " << *fTagIt );
                    KTagged::CloseTag( *fTagIt );
                }
                initmsg_debug( " >" << ret );
                initmsg_debug( "in path <" << aToken->GetPath() << "> in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom );
                fTags.clear();
                fTagStack.pop();
                return;
            }

            KProcessor::ProcessToken( aToken );
            return;
        }

        return;
    }

}

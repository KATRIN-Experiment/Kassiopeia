#include "KChattyProcessor.hh"
#include "KInitializationMessage.hh"

namespace katrin
{

    KChattyProcessor::KChattyProcessor()
    {
    }
    KChattyProcessor::~KChattyProcessor()
    {
    }

    void KChattyProcessor::ProcessToken( KBeginParsingToken* aToken)
    {
        initmsg( eNormal ) << "got a begin parsing token" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KBeginFileToken* aToken )
    {
        initmsg( eNormal ) << "got a begin file token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        initmsg( eNormal ) << "got a begin element token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
        initmsg( eNormal ) << "got a begin attribute token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        initmsg( eNormal ) << "got an attribute data token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
        initmsg( eNormal ) << "got an end attribute token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KMidElementToken* aToken )
    {
        initmsg( eNormal ) << "got a mid element token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KElementDataToken* aToken )
    {
        initmsg( eNormal ) << "got an element data token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KEndElementToken* aToken )
    {
        initmsg( eNormal ) << "got an end element token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KEndFileToken* aToken )
    {
        initmsg( eNormal ) << "got an end file token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KEndParsingToken* aToken)
    {
        initmsg( eNormal ) << "got an end parsing token" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KCommentToken* aToken )
    {
        initmsg( eNormal ) << "got a comment token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KChattyProcessor::ProcessToken( KErrorToken* aToken )
    {
        initmsg( eNormal ) << "got an error token <" << aToken->GetValue() << ">" << eom;
        initmsg_debug( "at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom);
        KProcessor::ProcessToken( aToken );
        return;
    }

}

#include "KIncludeProcessor.hh"
#include "KInitializationMessage.hh"

#include "KFile.h"
using katrin::KFile;

#include "KTextFile.h"
using katrin::KTextFile;

#include "KXMLTokenizer.hh"

#include <cstdlib>

namespace katrin
{

    KIncludeProcessor::KIncludeProcessor() :
        KProcessor(),
        fElementState( eElementInactive ),
        fAttributeState( eAttributeInactive ),
        fNames(),
        fPaths(),
        fBases()
    {
    }

    KIncludeProcessor::~KIncludeProcessor()
    {
    }


    void KIncludeProcessor::ProcessToken( KBeginElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            if( aToken->GetValue() == "include" )
            {
                fElementState = eActive;
                return;
            }
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            initmsg( eError ) << "got unknown element <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            return;
        }

        return;
    }

    void KIncludeProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {

        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( aToken->GetValue() == "name" )
            {
                fAttributeState = eName;
                return;
            }
            if( aToken->GetValue() == "path" )
            {
                fAttributeState = ePath;
                return;
            }
            if( aToken->GetValue() == "base" )
            {
                fAttributeState = eBase;
                return;
            }

            initmsg( eError ) << "got unknown attribute <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            return;
        }

        return;
    }

    void KIncludeProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( fAttributeState == eName )
            {
                fNames.push_back( aToken->GetValue() );
                fAttributeState = eAttributeComplete;
                return;
            }

            if( fAttributeState == ePath )
            {
                fPaths.push_back( aToken->GetValue() );
                fAttributeState = eAttributeComplete;
                return;
            }

            if( fAttributeState == eBase )
            {
                fBases.push_back( aToken->GetValue() );
                fAttributeState = eAttributeComplete;
                return;
            }
        }

        return;
    }

    void KIncludeProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            if( fAttributeState == eAttributeComplete )
            {
                fAttributeState = eAttributeInactive;
                return;
            }
        }

        return;
    }

    void KIncludeProcessor::ProcessToken( KMidElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eActive )
        {
            fElementState = eElementComplete;
            return;
        }

        return;
    }

    void KIncludeProcessor::ProcessToken( KElementDataToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            initmsg( eError ) << "got unknown element data <" << aToken->GetValue() << ">" << ret;
            initmsg( eError ) << "in path <" << aToken->GetPath() << "in file <" << aToken->GetFile() << "> at line <" << aToken->GetLine() << ">, column <" << aToken->GetColumn() << ">" << eom;
            return;
        }

        return;
    }

    void KIncludeProcessor::ProcessToken( KEndElementToken* aToken )
    {
        if( fElementState == eElementInactive )
        {
            KProcessor::ProcessToken( aToken );
            return;
        }

        if( fElementState == eElementComplete )
        {
            KTextFile* aFile = new KTextFile();
            aFile->SetDefaultPath( CONFIG_DEFAULT_DIR );

            vector< string >::iterator It;
            for( It = fNames.begin(); It != fNames.end(); It++ )
            {
                aFile->AddToNames( *It );
            }
            for( It = fPaths.begin(); It != fPaths.end(); It++ )
            {
                aFile->AddToPaths( *It );
            }
            for( It = fBases.begin(); It != fBases.end(); It++ )
            {
                aFile->AddToBases( *It );
            }

            if( aFile->Open( KFile::eRead ) == false )
            {
                delete aFile;

                initmsg << "unable to open file with names <";
                It = fNames.begin();
                while( It != fNames.end() )
                {
                    initmsg << *It;
                    It++;
                    if( It != fNames.end() )
                    {
                        initmsg << ",";
                    }
                }
                initmsg << "> and paths <";
                It = fPaths.begin();
                while( It != fPaths.end() )
                {
                    initmsg << *It;
                    It++;
                    if( It != fPaths.end() )
                    {
                        initmsg << ",";
                    }
                }
                initmsg << "> and bases <";
                It = fBases.begin();
                while( It != fBases.end() )
                {
                    initmsg << *It;
                    It++;
                    if( It != fBases.end() )
                    {
                        initmsg << ",";
                    }
                }
                initmsg( eError ) << ">" << eom;
            }

            fElementState = eElementInactive;
            fNames.clear();
            fPaths.clear();
            fBases.clear();

            KXMLTokenizer* aNewTokenizer = new KXMLTokenizer();
            aNewTokenizer->InsertBefore( GetFirstParent() );
            aNewTokenizer->ProcessFile( aFile );
            aNewTokenizer->Remove();

            delete aNewTokenizer;
            delete aFile;
        }

        return;
    }
}

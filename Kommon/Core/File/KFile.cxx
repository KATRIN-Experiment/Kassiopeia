#include "KFile.h"
#include "KFileMessage.h"

#include <cstdlib>

namespace katrin
{

    KFile::KFile() :
        fPaths(),
        fDefaultPath( "" ),
        fBases(),
        fDefaultBase( "" ),
        fNames(),
        fResolvedPath( "" ),
        fResolvedBase( "" ),
        fResolvedName( "" ),
        fState( eClosed )
    {
    }
    KFile::~KFile()
    {
    }

    void KFile::AddToPaths( const string& aPath )
    {
        fPaths.push_back( aPath );
        return;
    }
    void KFile::SetDefaultPath( const string& aPath )
    {
        fDefaultPath = aPath;
        return;
    }
    void KFile::AddToBases( const string& aBase )
    {
        fBases.push_back( aBase );
        return;
    }
    void KFile::SetDefaultBase( const string& aBase )
    {
        fDefaultBase = aBase;
        return;
    }
    void KFile::AddToNames( const string& aName )
    {
        fNames.push_back( aName );
        return;
    }

    const string& KFile::GetBase() const
    {
        return fResolvedBase;
    }
    const string& KFile::GetPath() const
    {
        return fResolvedPath;
    }
    const string& KFile::GetName() const
    {
        return fResolvedName;
    }

    bool KFile::Open( Mode aMode )
    {
        if( fState == eClosed )
        {
            string tFileName;

            //first look through explicit filenames
            vector< string >::iterator tNameIt;
            for( tNameIt = fNames.begin(); tNameIt != fNames.end(); tNameIt++ )
            {

                tFileName = *tNameIt;

                filemsg_debug( "attempting to open file at explicit name <" << *tNameIt << ">" << eom );

                if( OpenFileSubclass( tFileName, aMode ) == true )
                {
                    fResolvedPath = tFileName.substr( 0, tFileName.find_last_of( fDirectoryMark ) );
                    fResolvedBase = tFileName.substr( tFileName.find_last_of( fDirectoryMark ) + 1 );
                    fResolvedName = tFileName;

                    filemsg_debug( "successfully opened file <" << fResolvedName << ">" << eom );

                    fState = eOpen;
                    return true;
                }

            }

            //then look through explicit bases in explicit paths
            vector< string >::iterator tBaseIt;
            vector< string >::iterator tPathIt;
            for( tPathIt = fPaths.begin(); tPathIt != fPaths.end(); tPathIt++ )
            {
                for( tBaseIt = fBases.begin(); tBaseIt != fBases.end(); tBaseIt++ )
                {

                    tFileName = *tPathIt + fDirectoryMark + *tBaseIt;

                    filemsg_debug( "attempting to open file at explicit path and base <" << tFileName << ">" << eom );

                    if( OpenFileSubclass( tFileName, aMode ) == true )
                    {
                        fResolvedPath = tFileName.substr( 0, tFileName.find_last_of( fDirectoryMark ) );
                        fResolvedBase = tFileName.substr( tFileName.find_last_of( fDirectoryMark ) + 1 );
                        fResolvedName = tFileName;

                        filemsg_debug( "successfully opened file <" << fResolvedName << ">" << eom );

                        fState = eOpen;
                        return true;
                    }
                }
            }

            //then look through explicit bases in default path
            if (fDefaultPath.empty() == false){
				for( tBaseIt = fBases.begin(); tBaseIt != fBases.end(); tBaseIt++ )
				{

					tFileName = fDefaultPath + fDirectoryMark + *tBaseIt;

					filemsg_debug( "attempting to open file at default path and explicit base <" << tFileName << ">" << eom );

					if( OpenFileSubclass( tFileName, aMode ) == true )
					{
						fResolvedPath = tFileName.substr( 0, tFileName.find_last_of( fDirectoryMark ) );
						fResolvedBase = tFileName.substr( tFileName.find_last_of( fDirectoryMark ) + 1 );
						fResolvedName = tFileName;

						filemsg_debug( "successfully opened file <" << fResolvedName << ">" << eom );

						fState = eOpen;
						return true;
					}
				}
            }

            //then look through explicit paths with default base
            if (fDefaultBase.empty() == false){
				for( tPathIt = fPaths.begin(); tPathIt != fPaths.end(); tPathIt++ )
				{

					tFileName = *tPathIt + fDirectoryMark + fDefaultBase;

					filemsg_debug( "attempting to open file at explicit path and default base <" << tFileName << ">" << eom );

					if( OpenFileSubclass( tFileName, aMode ) == true )
					{
						fResolvedPath = tFileName.substr( 0, tFileName.find_last_of( fDirectoryMark ) );
						fResolvedBase = tFileName.substr( tFileName.find_last_of( fDirectoryMark ) + 1 );
						fResolvedName = tFileName;

						filemsg_debug( "successfully opened file <" << fResolvedName << ">" << eom );

						fState = eOpen;
						return true;
					}
				}
            }

            //finally, try the install defaults
            if( (fDefaultPath.empty() == false) && (fDefaultBase.empty() == false) )
            {
                tFileName = fDefaultPath + fDirectoryMark + fDefaultBase;

                filemsg_debug( "attempting to open file at default path and base <" << tFileName << ">" << eom );

                if( OpenFileSubclass( tFileName, aMode ) == true )
                {
                    fResolvedPath = tFileName.substr( 0, tFileName.find_last_of( fDirectoryMark ) );
                    fResolvedBase = tFileName.substr( tFileName.find_last_of( fDirectoryMark ) + 1 );
                    fResolvedName = tFileName;

                    filemsg_debug( "successfully opened file <" << fResolvedName << ">" << eom );

                    fState = eOpen;
                    return true;
                }
            }

            filemsg << "could not open file with the following specifications:" << ret;
            filemsg << "  paths:" << ret;
            for( tPathIt = fPaths.begin(); tPathIt != fPaths.end(); tPathIt++ )
            {
                filemsg << "    " << *tPathIt << ret;
            }
            filemsg << "  default path:" << ret;
            filemsg << "    " << fDefaultPath << ret;
            filemsg << "  bases:" << ret;
            for( tBaseIt = fBases.begin(); tBaseIt != fBases.end(); tBaseIt++ )
            {
                filemsg << "    " << *tBaseIt << ret;
            }
            filemsg << "  default base:" << ret;
            filemsg << "    " << fDefaultBase << ret;
            filemsg << "  names:" << ret;
            for( tNameIt = fNames.begin(); tNameIt != fNames.end(); tNameIt++ )
            {
                filemsg << "    " << *tNameIt << ret;
            }
            filemsg( eWarning ) << eom;

            return false;

        }
        return true;
    }

    bool KFile::IsOpen()
    {
        if( fState == eOpen )
        {
            return true;
        }
        return false;
    }

    bool KFile::Close()
    {
        if( fState == eOpen )
        {
            if( CloseFileSubclass() == true )
            {
                fResolvedPath = string( "" );
                fResolvedBase = string( "" );
                fResolvedName = string( "" );

                fState = eClosed;
                return true;
            }
            else
            {
                return false;
            }
        }
        return true;
    }
    bool KFile::IsClosed()
    {
        if( fState == eClosed )
        {
            return true;
        }
        return false;
    }

    const string KFile::fDirectoryMark = string( "/" );
    const string KFile::fExtensionMark = string( "." );

}

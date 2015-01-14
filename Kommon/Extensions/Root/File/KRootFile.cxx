#include "KRootFile.h"

#include "KFileMessage.h"

#include <cstdlib>

namespace katrin
{

    KRootFile::KRootFile() :
        fFile( NULL )
    {
    }
    KRootFile::~KRootFile()
    {
    }

    bool KRootFile::OpenFileSubclass( const string& aName, const Mode& aMode )
    {
        if( aMode == eRead )
        {
            fFile = new TFile( aName.c_str(), "READ" );
        }
        if( aMode == eWrite )
        {
            fFile = new TFile( aName.c_str(), "RECREATE" );
        }
        if( aMode == eAppend )
        {
            fFile = new TFile( aName.c_str(), "UPDATE" );
        }

        if( fFile->IsZombie() == true )
        {
            delete fFile;
            fFile = NULL;
            return false;
        }

        return true;
    }
    bool KRootFile::CloseFileSubclass()
    {
        if( fFile != NULL )
        {
            fFile->Close();
            delete fFile;
            fFile = NULL;

            return true;
        }
        return false;
    }

    TFile* KRootFile::File()
    {
        if( fState == eOpen )
        {
            return fFile;
        }
        filemsg( eError ) << "attempting to access file pointer of unopened file " << eom;
        return NULL;
    }

}

/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////                                                   /////
/////  BBBB   U   U  IIIII  L      DDDD   EEEEE  RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB   U   U    I    L      D   D  EE     RRRR   /////
/////  B   B  U   U    I    L      D   D  E      R   R  /////
/////  BBBB    UUU   IIIII  LLLLL  DDDD   EEEEE  R   R  /////
/////                                                   /////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////

namespace katrin
{

    static int sRootFileStructure =
        KRootFileBuilder::Attribute< string >( "path" ) +
        KRootFileBuilder::Attribute< string >( "default_path" ) +
        KRootFileBuilder::Attribute< string >( "base" ) +
        KRootFileBuilder::Attribute< string >( "default_base" ) +
        KRootFileBuilder::Attribute< string >( "name" );

}

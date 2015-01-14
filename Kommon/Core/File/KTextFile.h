#ifndef KTEXTFILE_H_
#define KTEXTFILE_H_

#include "KFile.h"

#include <fstream>
using std::fstream;
using std::ios_base;

#include <cstdlib>

namespace katrin
{

    class KTextFile :
        public KFile
    {
        public:
            KTextFile();
            virtual ~KTextFile();

        public:
            fstream* File();

        protected:
            virtual bool OpenFileSubclass( const string& aName, const Mode& aMode );
            virtual bool CloseFileSubclass();

        private:
            fstream* fFile;
    };

    inline KTextFile* CreateConfigTextFile( const string& aBase )
    {
        KTextFile* tFile = new KTextFile();
        tFile->SetDefaultPath( CONFIG_DEFAULT_DIR );
        tFile->SetDefaultBase( aBase );
        return tFile;
    }

    inline KTextFile* CreateScratchTextFile( const string& aBase )
    {
        KTextFile* tFile = new KTextFile();
        tFile->SetDefaultPath( SCRATCH_DEFAULT_DIR );
        tFile->SetDefaultBase( aBase );
        return tFile;
    }

    inline KTextFile* CreateDataTextFile( const string& aBase )
    {
        KTextFile* tFile = new KTextFile();
        tFile->SetDefaultPath( DATA_DEFAULT_DIR );
        tFile->SetDefaultBase( aBase );
        return tFile;
    }

    inline KTextFile* CreateOutputTextFile( const string& aBase )
    {
        KTextFile* tFile = new KTextFile();
        tFile->SetDefaultPath( OUTPUT_DEFAULT_DIR );
        tFile->SetDefaultBase( aBase );
        return tFile;
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

#include "KComplexElement.hh"

namespace katrin
{

    typedef KComplexElement< KTextFile > KTextFileBuilder;

    template< >
    inline bool KTextFileBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "path" )
        {
            aContainer->CopyTo( fObject, &KFile::AddToPaths );
            return true;
        }
        if( aContainer->GetName() == "default_path" )
        {
            aContainer->CopyTo( fObject, &KFile::SetDefaultPath );
            return true;
        }
        if( aContainer->GetName() == "base" )
        {
            aContainer->CopyTo( fObject, &KFile::AddToBases );
            return true;
        }
        if( aContainer->GetName() == "default_base" )
        {
            aContainer->CopyTo( fObject, &KFile::SetDefaultBase );
            return true;
        }
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KFile::AddToBases );
            return true;
        }
        return false;
    }

}

#endif

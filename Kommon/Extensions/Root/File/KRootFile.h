#ifndef KROOTFILE_H_
#define KROOTFILE_H_

#include "KFile.h"
#include "TFile.h"

#include <cstdlib>

namespace katrin
{

    class KRootFile :
        public KFile
    {
        public:
            static KRootFile* CreateScratchRootFile( const string& aBase );
            static KRootFile* CreateDataRootFile( const string& aBase );
            static KRootFile* CreateOutputRootFile( const string& aBase );

        public:
            KRootFile();
            virtual ~KRootFile();

        public:
            TFile* File();

        protected:
            virtual bool OpenFileSubclass( const string& aName, const Mode& aMode );
            virtual bool CloseFileSubclass();

        private:
            TFile* fFile;
    };

    inline KRootFile* KRootFile::CreateScratchRootFile( const string& aBase )
    {
        KRootFile* tFile = new KRootFile();
        tFile->SetDefaultPath( SCRATCH_DEFAULT_DIR );
        tFile->SetDefaultBase( aBase );
        return tFile;
    }

	inline KRootFile* KRootFile::CreateDataRootFile( const string& aBase )
	{
		KRootFile* tFile = new KRootFile();
		tFile->SetDefaultPath( DATA_DEFAULT_DIR );
		tFile->SetDefaultBase( aBase );
		return tFile;
	}

	inline KRootFile* KRootFile::CreateOutputRootFile( const string& aBase )
	{
		KRootFile* tFile = new KRootFile();
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

    typedef KComplexElement< KRootFile > KRootFileBuilder;

    template< >
    inline bool KRootFileBuilder::AddAttribute( KContainer* aContainer )
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

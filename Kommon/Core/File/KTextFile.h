#ifndef KTEXTFILE_H_
#define KTEXTFILE_H_

#include "KFile.h"

#include <fstream>

namespace katrin
{

class KTextFile : public KFile
{
  public:
    KTextFile();
    ~KTextFile() override;

  public:
    std::fstream* File();

  protected:
    bool OpenFileSubclass(const std::string& aName, const Mode& aMode) override;
    bool CloseFileSubclass() override;

  private:
    std::fstream* fFile;
};

inline KTextFile* CreateConfigTextFile(const std::string& aBase)
{
    auto* tFile = new KTextFile();
    tFile->SetDefaultPath(CONFIG_DEFAULT_DIR);
    tFile->SetDefaultBase(aBase);
    return tFile;
}

inline KTextFile* CreateScratchTextFile(const std::string& aBase)
{
    auto* tFile = new KTextFile();
    tFile->SetDefaultPath(SCRATCH_DEFAULT_DIR);
    tFile->SetDefaultBase(aBase);
    return tFile;
}

inline KTextFile* CreateDataTextFile(const std::string& aBase)
{
    auto* tFile = new KTextFile();
    tFile->SetDefaultPath(DATA_DEFAULT_DIR);
    tFile->SetDefaultBase(aBase);
    return tFile;
}

inline KTextFile* CreateOutputTextFile(const std::string& aBase)
{
    auto* tFile = new KTextFile();
    tFile->SetDefaultPath(OUTPUT_DEFAULT_DIR);
    tFile->SetDefaultBase(aBase);
    return tFile;
}

}  // namespace katrin

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

typedef KComplexElement<KTextFile> KTextFileBuilder;

template<> inline bool KTextFileBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "path") {
        aContainer->CopyTo(fObject, &KFile::AddToPaths);
        return true;
    }
    if (aContainer->GetName() == "default_path") {
        aContainer->CopyTo(fObject, &KFile::SetDefaultPath);
        return true;
    }
    if (aContainer->GetName() == "base") {
        aContainer->CopyTo(fObject, &KFile::AddToBases);
        return true;
    }
    if (aContainer->GetName() == "default_base") {
        aContainer->CopyTo(fObject, &KFile::SetDefaultBase);
        return true;
    }
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KFile::AddToBases);
        return true;
    }
    return false;
}

}  // namespace katrin

#endif

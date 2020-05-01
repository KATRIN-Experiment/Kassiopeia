#ifndef KROOTFILE_H_
#define KROOTFILE_H_

#include "KFile.h"

#include <TFile.h>

namespace katrin
{

class KRootFile : public KFile
{
  public:
    static KRootFile* CreateScratchRootFile(const std::string& aBase);
    static KRootFile* CreateDataRootFile(const std::string& aBase);
    static KRootFile* CreateOutputRootFile(const std::string& aBase);
    static KRootFile* CreateOutputRootFile(const std::string& aPath, const std::string& aBase);

  public:
    KRootFile();
    ~KRootFile() override;

  public:
    TFile* File();

  protected:
    bool OpenFileSubclass(const std::string& aName, const Mode& aMode) override;
    bool CloseFileSubclass() override;

  private:
    TFile* fFile;
};

inline KRootFile* KRootFile::CreateScratchRootFile(const std::string& aBase)
{
    auto* tFile = new KRootFile();
    tFile->SetDefaultPath(SCRATCH_DEFAULT_DIR);
    tFile->SetDefaultBase(aBase);
    return tFile;
}

inline KRootFile* KRootFile::CreateDataRootFile(const std::string& aBase)
{
    auto* tFile = new KRootFile();
    tFile->SetDefaultPath(DATA_DEFAULT_DIR);
    tFile->SetDefaultBase(aBase);
    return tFile;
}

inline KRootFile* KRootFile::CreateOutputRootFile(const std::string& aBase)
{
    auto* tFile = new KRootFile();
    tFile->SetDefaultPath(OUTPUT_DEFAULT_DIR);
    tFile->SetDefaultBase(aBase);
    return tFile;
}

inline KRootFile* KRootFile::CreateOutputRootFile(const std::string& aPath, const std::string& aBase)
{
    auto* tFile = new KRootFile();
    tFile->SetDefaultPath(aPath);
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

typedef KComplexElement<KRootFile> KRootFileBuilder;

template<> inline bool KRootFileBuilder::AddAttribute(KContainer* aContainer)
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

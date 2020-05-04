#include "KRootFile.h"

#include "KFileMessage.h"

using namespace std;

namespace katrin
{

KRootFile::KRootFile() : fFile(nullptr) {}
KRootFile::~KRootFile() {}

bool KRootFile::OpenFileSubclass(const string& aName, const Mode& aMode)
{
    if (aMode == eRead) {
        fFile = new TFile(aName.c_str(), "READ");
    }
    if (aMode == eWrite) {
        fFile = new TFile(aName.c_str(), "RECREATE");
    }
    if (aMode == eAppend) {
        fFile = new TFile(aName.c_str(), "UPDATE");
    }

    if (fFile->IsZombie() == true) {
        delete fFile;
        fFile = nullptr;
        return false;
    }

    return true;
}
bool KRootFile::CloseFileSubclass()
{
    if (fFile != nullptr) {
        fFile->Close();
        delete fFile;
        fFile = nullptr;

        return true;
    }
    return false;
}

TFile* KRootFile::File()
{
    if (fState == eOpen) {
        return fFile;
    }
    filemsg(eError) << "attempting to access file pointer of unopened file " << eom;
    return nullptr;
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

namespace katrin
{

STATICINT sRootFileStructure =
    KRootFileBuilder::Attribute<string>("path") + KRootFileBuilder::Attribute<string>("default_path") +
    KRootFileBuilder::Attribute<string>("base") + KRootFileBuilder::Attribute<string>("default_base") +
    KRootFileBuilder::Attribute<string>("name");

}

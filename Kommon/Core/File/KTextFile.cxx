#include "KTextFile.h"

#include "KFileMessage.h"

using namespace std;

namespace katrin
{

KTextFile::KTextFile() : fFile(nullptr) {}
KTextFile::~KTextFile() = default;

bool KTextFile::OpenFileSubclass(const string& aName, const Mode& aMode)
{
    if (aMode == eRead) {
        fFile = new fstream(aName.c_str(), ios_base::in);
    }
    if (aMode == eWrite) {
        fFile = new fstream(aName.c_str(), ios_base::out);
    }
    if (aMode == eAppend) {
        fFile = new fstream(aName.c_str(), ios_base::app);
    }

    if (fFile->fail() == true) {
        delete fFile;
        fFile = nullptr;
        return false;
    }

    return true;
}
bool KTextFile::CloseFileSubclass()
{
    if (fFile != nullptr) {
        fFile->close();
        delete fFile;
        fFile = nullptr;

        return true;
    }
    return false;
}

fstream* KTextFile::File()
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

STATICINT sTextFileStructure =
    //! a directory in which the file is looked for.
    KTextFileBuilder::Attribute<std::string>("path") +
    //! the directory used if the file cannot be opened with any ``path``.
    //! @deprecated has no effect: ``<messages>`` sets the default directory of its log file itself.
    KTextFileBuilder::Attribute<std::string>("default_path") +
    //! the name of the file (without directory).
    KTextFileBuilder::Attribute<std::string>("base") +
    //! the file name used if the file cannot be opened with any ``base``.
    //! @deprecated has no effect: ``<messages>`` sets the default name of its log file itself.
    KTextFileBuilder::Attribute<std::string>("default_base") +
    //! same as ``base``.
    //! @deprecated use ``base`` instead.
    KTextFileBuilder::Attribute<std::string>("name");

}

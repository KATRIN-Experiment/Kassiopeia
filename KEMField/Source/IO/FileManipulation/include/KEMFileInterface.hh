#ifndef KEMFILEINTERFACE_DEF
#define KEMFILEINTERFACE_DEF

#include "KEMFile.hh"
#include "KSAInputNode.hh"
#include "KSAOutputNode.hh"

#include <set>

namespace KEMField
{

/**
   * @class KEMFileInterface
   *
   * @brief A class for reading and writing KEMField files.
   *
   * KEMFileInterface is a class for reading and writing KEMField files.
   *
   * @author T.J. Corona
   */

class KEMFileInterface : public KEMFile
{
  public:
    static KEMFileInterface* GetInstance();

    template<class Readable>
    void FindByName(Readable& readable, const std::string& name, bool& result = KEMFileInterface::fNullResult,
                    std::string& resultFile = KEMFileInterface::fEmptyString);

    template<class Readable>
    void FindByHash(Readable& readable, const std::string& hash, bool& result = KEMFileInterface::fNullResult,
                    std::string& resultFile = KEMFileInterface::fEmptyString);

    template<class Readable>
    void FindByLabel(Readable& readable, const std::string& label, unsigned int index = 0,
                     bool& result = KEMFileInterface::fNullResult,
                     std::string& resultFile = KEMFileInterface::fEmptyString);

    template<class Readable>
    void FindByLabels(Readable& readable, const std::vector<std::string>& labels, unsigned int index = 0,
                      bool& result = KEMFileInterface::fNullResult,
                      std::string& resultFile = KEMFileInterface::fEmptyString);

    void ReadKSAFileFromActiveDirectory(KSAInputNode* node, const std::string& file_name,
                                        bool& result = KEMFileInterface::fNullResult) const;
    void SaveKSAFileToActiveDirectory(KSAOutputNode* node, const std::string& file_name,
                                      bool& result = KEMFileInterface::fNullResult, bool forceOverwrite = false) const;

    static void ReadKSAFile(KSAInputNode* node, const std::string& file_name,
                            bool& result = KEMFileInterface::fNullResult);
    static void SaveKSAFile(KSAOutputNode* node, const std::string& file_name,
                            bool& result = KEMFileInterface::fNullResult);

    unsigned int NumberWithLabel(const std::string& label) const;
    unsigned int NumberWithLabels(const std::vector<std::string>& labels) const;

    std::set<std::string> FileNamesWithLabels(const std::vector<std::string>& labels) const;
    std::set<std::string> FileList(std::string directory = "") const;
    std::set<std::string> CompleteFileList(std::string directory = "") const;

    static bool DirectoryExists(const std::string& directory);
    static bool CreateDirectory(const std::string& directory);
    static bool RemoveDirectory(const std::string& directory);
    bool RemoveFileFromActiveDirectory(const std::string& file_name);
    static bool DoesFileExist(const std::string& file_name);

    void ActiveDirectory(const std::string& directory);
    std::string ActiveDirectory() const
    {
        return fActiveDirectory;
    }

  private:
    KEMFileInterface();
    ~KEMFileInterface() override = default;

    static KEMFileInterface* fEMFileInterface;

    std::string fActiveDirectory;

    static std::string fEmptyString;
    static bool fNullResult;
};

template<class Readable>
void KEMFileInterface::FindByName(Readable& readable, const std::string& name, bool& result, std::string& resultFile)
{
    result = true;
    std::set<std::string> fileList = FileList();

    // move active file name to the front
    if (!fFileName.empty()) {
        fileList.erase(fFileName);
        fileList.insert(fFileName);
    }

    for (const auto& it : fileList) {
        if (HasElement(it, name)) {
            resultFile = it;
            return Read(it, readable, name);
        }
    }
    result = false;
}

template<class Readable>
void KEMFileInterface::FindByHash(Readable& readable, const std::string& hash, bool& result, std::string& resultFile)
{
    result = true;
    std::set<std::string> fileList = FileList();

    // move active file name to the front
    if (!fFileName.empty()) {
        fileList.erase(fFileName);
        fileList.insert(fFileName);
    }

    for (const auto& it : fileList) {
        if (HasElement(it, hash)) {
            resultFile = it;
            return ReadHashed(it, readable, hash);
        }
    }
    result = false;
}

template<class Readable>
void KEMFileInterface::FindByLabel(Readable& readable, const std::string& label, unsigned int index, bool& result,
                                   std::string& resultFile)
{
    result = true;
    std::set<std::string> fileList = FileList();

    // move active file name to the front
    if (!fFileName.empty()) {
        fileList.erase(fFileName);
        fileList.insert(fFileName);
    }

    for (const auto& it : fileList) {
        unsigned int nLabeled = NumberOfLabeled(it, label);
        if (index < nLabeled) {
            resultFile = it;
            return ReadLabeled(it, readable, label, index);
        }
        else
            index -= nLabeled;
    }
    result = false;
}

template<class Readable>
void KEMFileInterface::FindByLabels(Readable& readable, const std::vector<std::string>& labels, unsigned int index,
                                    bool& result, std::string& resultFile)
{
    result = true;
    std::set<std::string> fileList = FileList();

    // move active file name to the front
    if (!fFileName.empty()) {
        fileList.erase(fFileName);
        fileList.insert(fFileName);
    }

    for (const auto& it : fileList) {
        unsigned int nLabeled = NumberOfLabeled(it, labels);
        if (index < nLabeled) {
            resultFile = it;
            return ReadLabeled(it, readable, labels, index);
        }
        else
            index -= nLabeled;
    }
    result = false;
}
}  // namespace KEMField

#endif /* KEMFILEINTERFACE_DEF */

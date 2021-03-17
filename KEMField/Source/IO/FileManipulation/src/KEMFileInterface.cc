#include "KEMFileInterface.hh"

#include "KEMCoreMessage.hh"
#include "KSAStructuredASCIIHeaders.hh"

#include <cstdio>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#ifndef DEFAULT_SAVED_FILE_DIR
#define DEFAULT_SAVED_FILE_DIR "."
#endif /* !DEFAULT_SAVED_FILE_DIR */

using std::set;
using std::string;
using std::vector;

namespace KEMField
{
KEMFileInterface* KEMFileInterface::fEMFileInterface = nullptr;
string KEMFileInterface::fEmptyString = "";
bool KEMFileInterface::fNullResult = false;

KEMFileInterface::KEMFileInterface()
{
    ActiveDirectory(DEFAULT_SAVED_FILE_DIR);
}

/**
   * Interface to accessing KEMFileInterface.
   */
KEMFileInterface* KEMFileInterface::GetInstance()
{
    if (fEMFileInterface == nullptr)
        fEMFileInterface = new KEMFileInterface();
    return fEMFileInterface;
}

unsigned int KEMFileInterface::NumberWithLabel(const string& label) const
{
    unsigned int value = 0;
    set<string> fileList = FileList();

    for (const auto& it : fileList)
        value += NumberOfLabeled(it, label);
    return value;
}

unsigned int KEMFileInterface::NumberWithLabels(const vector<string>& labels) const
{
    unsigned int value = 0;
    set<string> fileList = FileList();

    for (const auto& it : fileList)
        value += NumberOfLabeled(it, labels);
    return value;
}

set<string> KEMFileInterface::FileNamesWithLabels(const vector<string>& labels) const
{
    set<string> fileList = FileList();
    set<string> labeledFileList;

    for (const auto& it : fileList) {
        if (NumberOfLabeled(it, labels)) {
            labeledFileList.insert(it);
        }
    }
    return labeledFileList;
}


set<string> KEMFileInterface::FileList(string directory) const
{
    if (directory.empty())
        directory = fActiveDirectory;

    set<string> fileList;

    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(directory.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            string entry(ent->d_name);
            if (entry.find_last_of('.') == string::npos)
                continue;
            string suffix = entry.substr(entry.find_last_of('.'), string::npos);
            if (suffix == fStreamer.GetFileSuffix())
                fileList.insert(directory + "/" + entry);
        }
        closedir(dir);
    }
    return fileList;
}

set<string> KEMFileInterface::CompleteFileList(string directory) const
{
    if (directory.empty())
        directory = fActiveDirectory;

    set<string> fileList;

    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(directory.c_str())) != nullptr) {
        while ((ent = readdir(dir)) != nullptr) {
            string entry(ent->d_name);
            if (entry.find_last_of('.') == string::npos)
                continue;
            fileList.insert(directory + "/" + entry);
        }
        closedir(dir);
    }
    return fileList;
}

void KEMFileInterface::ActiveDirectory(const string& directory)
{
    if (!DirectoryExists(directory))
        CreateDirectory(directory);
    if (DirectoryExists(directory))
        fActiveDirectory = directory;
    else
        kem_cout(eError) << "Cannot access directory " << directory << eom;
}

bool KEMFileInterface::DirectoryExists(const string& directory)
{
    DIR* dir;
    if ((dir = opendir(directory.c_str())) != nullptr) {
        closedir(dir);
        return true;
    }
    return false;
}

bool KEMFileInterface::CreateDirectory(const string& directory)
{
    return mkdir(directory.c_str(), S_IRWXU);
}

bool KEMFileInterface::RemoveDirectory(const string& directory)
{
    return rmdir(directory.c_str());
}

bool KEMFileInterface::RemoveFileFromActiveDirectory(const string& file_name)
{
    string full_file_name = fActiveDirectory + "/" + file_name;
    return std::remove(full_file_name.c_str());
}

bool KEMFileInterface::DoesFileExist(const std::string& file_name)
{
    std::string full_file_name = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + file_name;
    std::set<std::string> file_list = KEMFileInterface::GetInstance()->CompleteFileList();
    for (const auto& it : file_list) {
        if (full_file_name == it) {
            return true;
        };
    }
    return false;
}

void KEMFileInterface::ReadKSAFile(KSAInputNode* node, const string& file_name, bool& result)
{
    result = false;
    KSAFileReader reader;

    reader.SetFileName(std::move(file_name));
    if (reader.Open()) {
        KSAInputCollector collector;
        collector.SetFileReader(&reader);
        collector.ForwardInput(node);
        if (node->HasData()) {
            result = true;
        }
        reader.Close();
    }
    return;
}

void KEMFileInterface::ReadKSAFileFromActiveDirectory(KSAInputNode* node, const string& file_name, bool& result) const
{
    std::string full_file_name = ActiveDirectory() + "/";
    full_file_name += file_name;
    result = false;
    KSAFileReader reader;

    reader.SetFileName(full_file_name);
    if (reader.Open()) {
        KSAInputCollector collector;
        collector.SetFileReader(&reader);
        collector.ForwardInput(node);
        if (node->HasData()) {
            result = true;
        }
        reader.Close();
    }
    return;
}

void KEMFileInterface::SaveKSAFile(KSAOutputNode* node, const string& file_name, bool& result)
{
    //now stream the data out to file
    KSAFileWriter writer;
    KSAOutputCollector collector;
    collector.SetUseTabbingFalse();
    writer.SetFileName(std::move(file_name));

    if (writer.Open()) {
        collector.SetFileWriter(&writer);
        collector.CollectOutput(node);
        writer.Close();
        result = true;
        return;
    }

    result = false;
    //failure to open file for writing
}

void KEMFileInterface::SaveKSAFileToActiveDirectory(KSAOutputNode* node, const string& file_name, bool& result,
                                                    bool forceOverwrite) const
{
    result = false;
    set<string> fileList = CompleteFileList();
    std::string full_file_name = ActiveDirectory() + "/" + file_name;

    for (const auto& it : fileList) {
        if (it == full_file_name) {
            if (!forceOverwrite) {
                //file already exists, and we do not want to overwrite it
                result = false;
                return;
            }
        }
    }

    //file doesn't already exist or we can overwrite it, safe to write
    //now stream the data out to file
    KSAFileWriter writer;
    KSAOutputCollector collector;
    collector.SetUseTabbingFalse();
    writer.SetFileName(full_file_name);

    if (writer.Open()) {
        collector.SetFileWriter(&writer);
        collector.CollectOutput(node);
        writer.Close();
        result = true;
        return;
    }

    result = false;
    //failure to open file for writing
}


}  // namespace KEMField

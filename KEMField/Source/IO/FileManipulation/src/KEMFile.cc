#include "KEMFile.hh"
#include "KEMCout.hh"  // for Inspect()
#include "KEMCoreMessage.hh"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <utility>

#ifndef DEFAULT_SAVED_FILE_DIR
#define DEFAULT_SAVED_FILE_DIR "."
#endif /* !DEFAULT_SAVED_FILE_DIR */

namespace KEMField
{
KEMFile::KEMFile()
{
    time_t t = time(nullptr);
    struct tm* now = localtime(&t);
    std::stringstream s;
    s << DEFAULT_SAVED_FILE_DIR << "/KEM_" << (now->tm_year + 1900) << '-'
      << std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << '-'
      << std::setfill('0') << std::setw(2) << now->tm_mday << "_"
      << std::setfill('0') << std::setw(2) << now->tm_hour << "-"
      << std::setfill('0') << std::setw(2) << now->tm_min << "-"
      << std::setfill('0') << std::setw(2) << now->tm_sec
      << ".kbd";
    fFileName = s.str();
}

KEMFile::KEMFile(const std::string& fileName) : fFileName(std::move(fileName)) {}

KEMFile::~KEMFile() = default;

void KEMFile::Inspect(const std::string& fileName) const
{
    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);

        fStreamer >> key;
        KEMField::cout << key << KEMField::endl;

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
    }

    fStreamer.close();
}

bool KEMFile::HasElement(const std::string& fileName, const std::string& name) const
{
    kem_cout_debug("Checking for a label in file <" << fileName << ">" << eom);

    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool hasElement = false;
    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        if (key.fObjectName == name || key.fObjectHash == name) {
            hasElement = true;
            break;
        }

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
    }

    fStreamer.close();
    return hasElement;
}

bool KEMFile::HasLabeled(const std::string& fileName, const std::vector<std::string>& labels) const
{
    kem_cout_debug("Checking for " << labels.size() << " labels in file <" << fileName << ">" << eom);

    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool hasLabeled = false;
    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        hasLabeled = true;
        for (auto& label : labels) {
            auto it2 = std::find(key.fLabels.begin(), key.fLabels.end(), label);
            if (it2 == key.fLabels.end()) {
                hasLabeled = false;
                break;
            }
        }
        if (hasLabeled)
            break;

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
    }

    fStreamer.close();
    return hasLabeled;
}

unsigned int KEMFile::NumberOfLabeled(const std::string& fileName, const std::string& label) const
{
    kem_cout_debug("Checking for a label in file <" << fileName << ">" << eom);

    unsigned int value = 0;
    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        for (auto& l : key.fLabels) {
            if (l == label) {
                value++;
                break;
            }
        }

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
    }

    fStreamer.close();
    return value;
}

unsigned int KEMFile::NumberOfLabeled(const std::string& fileName, const std::vector<std::string>& labels) const
{
    kem_cout_debug("Checking for " << labels.size() << " labels in file <" << fileName << ">" << eom);

    unsigned int value = 0;
    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        bool hasLabeled = true;
        for (auto& label : labels) {
            auto it2 = std::find(key.fLabels.begin(), key.fLabels.end(), label);
            if (it2 == key.fLabels.end()) {
                hasLabeled = false;
                break;
            }
        }
        if (hasLabeled)
            value++;

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
    }

    fStreamer.close();
    return value;
}

std::vector<std::string> KEMFile::LabelsForElement(const std::string& fileName, const std::string& name) const
{
    kem_cout_debug("Reading element labels from file <" << fileName << ">" << eom);

    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        if (key.fObjectName == name)
            break;

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
        key.clear();
    }

    fStreamer.close();
    return key.fLabels;
}

bool KEMFile::ElementHasLabel(const std::string& fileName, const std::string& name, const std::string& label) const
{
    std::vector<std::string> labels = LabelsForElement(std::move(fileName), std::move(name));
    for (auto& it : labels)
        if (it == label)
            return true;
    return false;
}

bool KEMFile::FileExists(const std::string& fileName)
{
    struct stat fileInfo;
    int fileStat;

    fileStat = stat(fileName.c_str(), &fileInfo);
    if (fileStat == 0)
        return true;
    else
        return false;
}

KEMFile::Key KEMFile::KeyForElement(const std::string& fileName, const std::string& name)
{
    kem_cout_debug("Reading an element key from file <" << fileName << ">" << eom);

    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        if (key.fObjectName == name)
            break;

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
        key.clear();
    }

    fStreamer.close();
    return key;
}

KEMFile::Key KEMFile::KeyForHashed(const std::string& fileName, const std::string& hash)
{
    kem_cout_debug("Reading a hashed key from file <" << fileName << ">" << eom);

    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        if (key.fObjectHash == hash)
            break;

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
        key.clear();
    }

    fStreamer.close();
    return key;
}

KEMFile::Key KEMFile::KeyForLabeled(const std::string& fileName, const std::string& label, unsigned int index)
{
    kem_cout_debug("Reading a labeled key from file <" << fileName << ">" << eom);

    unsigned int index_ = 0;
    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool found = false;
    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;
        for (auto& l : key.fLabels)
            if (l == label) {
                if (index != index_) {
                    index_++;
                    break;
                }
                else
                    found = true;
            }
        if (found)
            break;

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
            kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
           break;
        }
        key.clear();
    }

    fStreamer.close();
    return key;
}
}  // namespace KEMField

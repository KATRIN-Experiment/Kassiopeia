#ifndef KEMFILE_DEF
#define KEMFILE_DEF

#include "KBinaryDataStreamer.hh"
#include "KEMCoreMessage.hh"
#include "KMD5HashGenerator.hh"
#include "KStreamedSizeOf.hh"

#include <algorithm>
#include <cstdarg>
#include <cstdlib>
#include <sstream>
#include <vector>

namespace KEMField
{

/**
   * @class KEMFile
   *
   * @brief A class for reading and writing KEMField files.
   *
   * KEMFile is a class for reading and writing files, allowing for random
   * access using a key system.
   *
   * @author T.J. Corona
   */

class KEMFile
{
  public:
    KEMFile();
    KEMFile(const std::string& fileName);
    virtual ~KEMFile();

    void ActiveFile(const std::string& fileName)
    {
        fFileName = fileName;
    }
    std::string GetActiveFileName() const
    {
        return fFileName;
    }

    template<class Writable>
    void Write(const std::string&, const Writable&, const std::string&, const std::vector<std::string>&);

    template<class Writable> void Write(const std::string&, const Writable&, const std::string&);

    template<class Writable> void Write(const std::string&, const Writable&, const std::string&, const std::string&);

    template<class Writable> void Write(const Writable& w, const std::string& name)
    {
        Write(fFileName, w, name);
    }

    template<class Writable> void Write(const Writable& w, const std::string& name, const std::string& label)
    {
        Write(fFileName, w, name, label);
    }

    template<class Writable>
    void Write(const Writable& w, const std::string& name, const std::vector<std::string>& labels)
    {
        Write(fFileName, w, name, labels);
    }

    template<class Readable> void Read(const std::string&, Readable&, const std::string&);

    template<class Readable> void Read(Readable& r, const std::string& s)
    {
        Read(fFileName, r, s);
    }

    template<class Readable> void ReadHashed(const std::string&, Readable&, const std::string&);

    template<class Readable>
    void ReadLabeled(const std::string&, Readable&, const std::string&, unsigned int index = 0);

    template<class Readable>
    void ReadLabeled(const std::string&, Readable&, const std::vector<std::string>&, unsigned int index = 0);

    template<class Writable> void Overwrite(const std::string&, const Writable&, const std::string&);

    template<class Writable> void Overwrite(const Writable& w, const std::string& s)
    {
        Overwrite(fFileName, w, s);
    }

    void Inspect(const std::string&) const;

    bool HasElement(const std::string&, const std::string&) const;
    bool HasLabeled(const std::string&, const std::vector<std::string>&) const;
    unsigned int NumberOfLabeled(const std::string&, const std::string&) const;
    unsigned int NumberOfLabeled(const std::string&, const std::vector<std::string>&) const;

    std::vector<std::string> LabelsForElement(const std::string&, const std::string&) const;
    bool ElementHasLabel(const std::string&, const std::string&, const std::string&) const;

    static bool FileExists(const std::string&);

    std::string GetFileSuffix() const
    {
        return fStreamer.GetFileSuffix();
    }

  protected:
    std::string fFileName;

    KMD5HashGenerator fHashGenerator;

    mutable KBinaryDataStreamer fStreamer;

    struct Key
    {
        Key() = default;
        ~Key() = default;

        static std::string Name()
        {
            return "Key";
        }

        void clear()
        {
            fObjectName.clear();
            fClassName.clear();
            fObjectHash.clear();
            fLabels.clear();
            fObjectLocation = fObjectSize = 0;
        }

        size_t NextKey() const
        {
            return fObjectLocation + fObjectSize;
        }

        template<typename Stream> friend Stream& operator>>(Stream& s, Key& k)
        {
            s.PreStreamInAction(k);
            s >> k.fObjectName;
            s >> k.fClassName;
            s >> k.fObjectHash;
            unsigned int size;
            s >> size;
            k.fLabels.resize(size);
            for (unsigned int i = 0; i < size; i++)
                s >> k.fLabels[i];
            s >> k.fObjectLocation;
            s >> k.fObjectSize;
            s.PostStreamInAction(k);
            return s;
        }

        template<typename Stream> friend Stream& operator<<(Stream& s, const Key& k)
        {
            s.PreStreamOutAction(k);
            s << k.fObjectName;
            s << k.fClassName;
            s << k.fObjectHash;
            s << (unsigned int) (k.fLabels.size());
            for (const auto& label : k.fLabels)
                s << label;
            s << k.fObjectLocation;
            s << k.fObjectSize;
            s.PostStreamOutAction(k);
            return s;
        }

        std::string fObjectName;
        std::string fClassName;
        std::string fObjectHash;
        std::vector<std::string> fLabels;
        size_t fObjectLocation;
        size_t fObjectSize;
    };

    Key KeyForElement(const std::string&, const std::string&);
    Key KeyForHashed(const std::string&, const std::string&);
    Key KeyForLabeled(const std::string&, const std::string&, unsigned int index = 0);
};

template<class Writable>
void KEMFile::Write(const std::string& fileName, const Writable& writable, const std::string& name)
{
    std::vector<std::string> labels(0);
    Write<Writable>(fileName, writable, name, labels);
}

template<class Writable>
void KEMFile::Write(const std::string& fileName, const Writable& writable, const std::string& name,
                    const std::string& label)
{
    std::vector<std::string> labels;
    labels.push_back(label);
    Write<Writable>(fileName, writable, name, labels);
}

template<class Writable>
void KEMFile::Write(const std::string& fileName, const Writable& writable, const std::string& name,
                    const std::vector<std::string>& labels)
{

    if (!FileExists(fileName)) {
        fStreamer.open(fileName, "overwrite");
        fStreamer.close();
    }
    else {
        fStreamer.open(fileName, "read");

        // Pull and check the keys sequentially
        Key key;

        size_t readPoint = 0;
        fStreamer.Stream().seekg(0, fStreamer.Stream().end);
        size_t end = fStreamer.Stream().tellg();
        fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

        bool duplicateName = false;

        while (fStreamer.Stream().good() && readPoint < end) {
            fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
            fStreamer >> key;

            if (key.fObjectName == name)
                duplicateName = true;

            size_t lastReadPoint = readPoint;
            readPoint = key.NextKey();
            if (readPoint <= lastReadPoint) {
                kem_cout(eError) << "File <" << fileName << "> could not be read (" << end-lastReadPoint << " bytes remain)" << ret;
                kem_cout << "Stored data in the file might be corrupted. You need to manually remove this file to continue." << eom;
               break;
            }
        }

        fStreamer.close();

        if (duplicateName) {
            kem_cout(eWarning) << "Element <" << name << "> already exists in file <" << fileName << ">." << eom;
            return;
        }
    }

    fStreamer.open(fileName, "update");

    // First, create a key for our object
    Key key;
    key.fObjectName = name;
    key.fClassName = Writable::Name();

    key.fObjectHash = fHashGenerator.GenerateHash(writable);

    for (auto& label : labels)
        key.fLabels.push_back(label);

    // Write the incomplete key into the stream
    fStreamer.Stream().seekp(0, fStreamer.Stream().end);
    size_t keyLocation = fStreamer.Stream().tellp();
    fStreamer << key;

    // Write the object into the stream
    size_t objectLocation = fStreamer.Stream().tellp();
    fStreamer << writable;
    size_t objectEnd = fStreamer.Stream().tellp();

    // Complete the key information, and overwrite the key
    key.fObjectLocation = objectLocation;
    key.fObjectSize = objectEnd - objectLocation;

    fStreamer.close();
    fStreamer.open(fileName, "modify");

    fStreamer.Stream().seekp(keyLocation, fStreamer.Stream().beg);
    fStreamer << key;

    fStreamer.close();
}

template<class Writable>
void KEMFile::Overwrite(const std::string& fileName, const Writable& writable, const std::string& name)
{
    if (!FileExists(fileName)) {
        kem_cout(eError) << "Cannot open file <" << fileName << ">." << eom;
        return;
    }

    kem_cout_debug("Overwriting an element in file <" << fileName << ">" << eom);

    // First, we find the key associated with our object
    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool elementFound = false;

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;

        if (key.fObjectName == name) {
            if (key.fClassName != Writable::Name()) {
                kem_cout(eInfo) << "Element <" << name << "> is stored as a " << key.fClassName << eom;
                break;
            }

            elementFound = true;
            break;
        }

        size_t lastReadPoint = readPoint;
        readPoint = key.NextKey();
        if (readPoint <= lastReadPoint) {
            kem_cout(eError) << "Corrupted file <" << fileName << ">, " << end-lastReadPoint << " bytes remain" << eom;
            break;
        }
    }

    fStreamer.close();

    if (!elementFound) {
        if (readPoint >= end)
            kem_cout(eWarning) << "Element <" << name << "> could not be located in file <" << fileName << ">." << eom;
        return;
    }

    // check to make sure the objects are the same size
    KStreamedSizeOf sizeOf;
    if (sizeOf(writable) != key.fObjectSize) {
        kem_cout(eError) << "Cannot overwrite element <" << name << "> with an instance of " << key.fClassName
                         << " of a different size." << eom;
    }

    // overwrite the modified element
    fStreamer.open(fileName, "modify");

    fStreamer.Stream().seekp(key.fObjectLocation, fStreamer.Stream().beg);
    fStreamer << writable;

    fStreamer.close();
}

template<class Readable>
void KEMFile::ReadHashed(const std::string& fileName, Readable& readable, const std::string& hash)
{
    if (!FileExists(fileName)) {
        kem_cout(eError) << "Cannot open file <" << fileName << ">." << eom;
        return;
    }

    kem_cout_debug("Reading a hashed element from file <" << fileName << ">" << eom);

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

        if (key.fObjectHash == hash) {
            if (key.fClassName != Readable::Name()) {
                kem_cout(eInfo) << "Element with hash <" << hash << "> is stored as a " << key.fClassName << eom;
            }
            else {
                fStreamer.Stream().seekg(key.fObjectLocation, fStreamer.Stream().beg);
                fStreamer >> readable;
            }
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
}

template<class Readable>
void KEMFile::ReadLabeled(const std::string& fileName, Readable& readable, const std::string& label, unsigned int index)
{
    if (!FileExists(fileName)) {
        kem_cout(eError) << "Cannot open file <" << fileName << ">." << eom;
        return;
    }

    kem_cout_debug("Reading a labeled element from file <" << fileName << ">" << eom);

    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    unsigned int index_ = 0;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool found = false;

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;

        for (auto it = key.fLabels.begin(); it != key.fLabels.end(); ++it) {
            if (*it == label) {
                // first, check if this is the correct index
                if (index != index_) {
                    index_++;
                    break;
                }
                else {
                    if (key.fClassName != Readable::Name())
                        kem_cout(eInfo) << "Label <" << label << "> is assigned to a " << key.fClassName << eom;
                    else {
                        fStreamer.Stream().seekg(key.fObjectLocation, fStreamer.Stream().beg);
                        fStreamer >> readable;
                    }
                    found = true;
                    break;
                }
            }
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
    }
    fStreamer.close();
}

template<class Readable>
void KEMFile::ReadLabeled(const std::string& fileName, Readable& readable, const std::vector<std::string>& labels,
                          unsigned int index)
{
    if (!FileExists(fileName)) {
        kem_cout(eError) << "Cannot open file <" << fileName << ">." << eom;
        return;
    }

    kem_cout_debug("Reading " << labels.size() << " labeled elements from file <" << fileName << ">" << eom);

    fStreamer.open(fileName, "read");

    // Pull and check the keys sequentially
    Key key;

    unsigned int index_ = 0;

    size_t readPoint = 0;
    fStreamer.Stream().seekg(0, fStreamer.Stream().end);
    size_t end = fStreamer.Stream().tellg();
    fStreamer.Stream().seekg(0, fStreamer.Stream().beg);

    bool found = false;

    while (fStreamer.Stream().good() && readPoint < end) {
        fStreamer.Stream().seekg(readPoint, fStreamer.Stream().beg);
        fStreamer >> key;

        found = true;
        for (auto& label : labels) {
            auto it2 = std::find(key.fLabels.begin(), key.fLabels.end(), label);
            if (it2 == key.fLabels.end()) {
                found = false;
                break;
            }
        }
        if (found) {
            // first, check if this is the correct index
            if (index != index_) {
                index_++;
            }
            else {
                if (key.fClassName != Readable::Name()) {
                    kem_cout(eInfo) << "Label set is assigned to a " << key.fClassName << eom;
                }
                else {
                    fStreamer.Stream().seekg(key.fObjectLocation, fStreamer.Stream().beg);
                    fStreamer >> readable;
                }
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
}

template<class Readable> void KEMFile::Read(const std::string& fileName, Readable& readable, const std::string& name)
{
    if (!FileExists(fileName)) {
        kem_cout << "Cannot open file <" << fileName << ">." << eom;
        return;
    }

    kem_cout_debug("Reading an element from file <" << fileName << ">" << eom);

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

        if (key.fObjectName == name) {
            if (key.fClassName != Readable::Name()) {
                kem_cout() << "Element <" << name << "> is stored as a " << key.fClassName << eom;
                break;
            }

            fStreamer.Stream().seekg(key.fObjectLocation, fStreamer.Stream().beg);
            fStreamer >> readable;
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
}

}  // namespace KEMField

#endif /* KEMFILE_DEF */

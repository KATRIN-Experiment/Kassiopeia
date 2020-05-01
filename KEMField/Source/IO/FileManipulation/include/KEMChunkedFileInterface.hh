#ifndef KEMChunkedFileInterface_HH__
#define KEMChunkedFileInterface_HH__

#include "KEMFileInterface.hh"

#include <cstdio>
#include <sstream>
#include <string>

namespace KEMField
{


/*
*
*@file KEMChunkedFileInterface.hh
*@class KEMChunkedFileInterface
*@brief This class is meant for fast file access by repeatedly writing/reading
* buffered chunks of binary data to/from a file in a continuous way
* The file is expected to contain entirely homogeneous data (all same type of object)
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Apr 28 09:16:29 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KEMChunkedFileInterface
{
  public:
    KEMChunkedFileInterface()
    {
        fFile = nullptr;
        fFileName = "";
        fFilePath = "";
    };

    virtual ~KEMChunkedFileInterface(){};

    //check if file exists in order to open it, or so we can avoid overwriting it
    bool DoesFileExist(std::string file_name)
    {
        std::string full_file_name = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + file_name;
        std::set<std::string> file_list = KEMFileInterface::GetInstance()->CompleteFileList();
        for (auto it = file_list.begin(); it != file_list.end(); ++it) {
            if (full_file_name == *it) {
                return true;
            };
        }
        return false;
    }

    bool OpenFileForWriting(std::string file_name)
    {
        fFileName = file_name;
        fFilePath = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFileName;

        FILE* pFile = fopen(fFilePath.c_str(), "wb");

        if (pFile == nullptr) {
            fFile = nullptr;
            return false;
        }

        fFile = pFile;
        return true;
    }

    bool OpenFileForReading(std::string file_name)
    {
        fFileName = file_name;
        fFilePath = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + fFileName;

        FILE* pFile = fopen(fFilePath.c_str(), "rb");
        if (pFile == nullptr) {
            fFile = nullptr;
            return false;
        }
        else {
            fFile = pFile;
            return true;
        }
    }

    template<typename ObjectType> size_t Write(size_t n_objects, const ObjectType* obj_arr)
    {
        //write the buffer out to file
        if (fFile != nullptr) {
            size_t n_objects_written = fwrite(obj_arr, sizeof(ObjectType), n_objects, fFile);
            return n_objects_written;
        }
        return 0;
    }

    template<typename ObjectType> size_t Read(size_t n_objects, ObjectType* obj_arr)
    {
        //read buffer in from file
        if (fFile != nullptr) {
            size_t n_objects_read = fread(obj_arr, sizeof(ObjectType), n_objects, fFile);
            return n_objects_read;
        }
        return 0;
    }

    void CloseFile()
    {
        if (fFile != nullptr) {
            fclose(fFile);
        }
        fFile = nullptr;
    }

  private:
    std::string fFilePath;
    std::string fFileName;
    FILE* fFile;
};


}  // namespace KEMField


#endif /*KEMChunkedFileInterface_H__ */

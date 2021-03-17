#ifndef KEMSparseMatrixFileInterface_HH__
#define KEMSparseMatrixFileInterface_HH__

#ifndef KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB
#define KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB 128  //size of buffer in megabytes
#endif

#include "KEMFileInterface.hh"

#include <cstdio>
#include <sstream>
#include <string>

namespace KEMField
{


/*
*
*@file KEMSparseMatrixFileInterface.hh
*@class KEMSparseMatrixFileInterface
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Apr 28 09:16:29 EDT 2014 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KEMSparseMatrixFileInterface
{
  public:
    KEMSparseMatrixFileInterface()
    {
        fPrefix = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + std::string("SparseMatrix_");
        fPredicate = std::string(".bin");
        fBufferSize = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB;
        fBufferSize *= sizeof(double) * 1024 * 1024;
    };

    virtual ~KEMSparseMatrixFileInterface() = default;
    ;


    virtual void SetFilePrefix(const std::string& file_prefix)
    {
        fPrefix = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + file_prefix;
    }

    virtual void SetFilePredicate(const std::string& file_predicate)
    {
        fPredicate = file_predicate;
    }


    virtual bool DoesSectionExist(unsigned int section)
    {
        std::stringstream ss;
        ss << fPrefix;
        ss << section;
        ss << fPredicate;

        std::string name = ss.str();

        std::set<std::string> file_list = KEMFileInterface::GetInstance()->CompleteFileList();

        for (const auto& it : file_list) {
            if (name == it) {
                return true;
            };
        }

        return false;
    }

    virtual void WriteMatrixElements(unsigned int section, const double* matrix_elements) const
    {
        //write the buffer out to file
        FILE* pFile;

        std::stringstream ss;
        ss << fPrefix;
        ss << section;
        ss << fPredicate;

        pFile = fopen(ss.str().c_str(), "wb");

        fwrite(matrix_elements, sizeof(double), fBufferSize, pFile);
        fclose(pFile);
    }

    virtual void ReadMatrixElements(unsigned int section, double* matrix_elements) const
    {
        //read buffer in from file
        FILE* pFile;

        std::stringstream ss;
        ss << fPrefix;
        ss << section;
        ss << fPredicate;

        pFile = fopen(ss.str().c_str(), "rb");

        size_t __attribute__((__unused__)) unused = fread(matrix_elements, sizeof(double), fBufferSize, pFile);
        fclose(pFile);
    }


  private:
    size_t fBufferSize;
    std::string fPrefix;
    std::string fPredicate;
};


}  // namespace KEMField


#endif /* KEMSparseMatrixFileInterface_H__ */

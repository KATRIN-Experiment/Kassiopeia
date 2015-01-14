#ifndef KEMSparseMatrixFileInterface_HH__
#define KEMSparseMatrixFileInterface_HH__

#ifndef KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB
    #define KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB 2048 //size of buffer in megabytes
#endif

#define KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE_MB*131072 //size of buffer in number of doubles

#include "KEMFileInterface.hh"

#include <string>
#include <sstream>
#include <stdio.h>

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
            fBufferSize = KEMFIELD_SPARSE_MATRIX_BUFFER_SIZE;
        };

        virtual ~KEMSparseMatrixFileInterface(){};


        virtual void SetFilePrefix(std::string file_prefix)
        {
            fPrefix = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + file_prefix;
        }

        virtual void SetFilePredicate(std::string file_predicate)
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

            std::set< std::string > file_list = KEMFileInterface::GetInstance()->CompleteFileList();

            for(std::set<std::string>::iterator it=file_list.begin(); it!=file_list.end(); ++it)
            {
                if(name == *it){return true;};
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

            fread(matrix_elements, sizeof(double), fBufferSize, pFile);
            fclose(pFile);
        }


    private:

        size_t fBufferSize;
        std::string fPrefix;
        std::string fPredicate;

};


}


#endif /* KEMSparseMatrixFileInterface_H__ */

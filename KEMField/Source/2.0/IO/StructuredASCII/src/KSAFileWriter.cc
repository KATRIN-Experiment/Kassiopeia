#include "KSAFileWriter.hh"

#include <iostream>

#include <assert.h>

#include "miniz.c"

namespace KEMField{

KSAFileWriter::KSAFileWriter():
fIsOpen(false),
fUseCompression(false),
fCompressionLevel(9)
{
    fUsedSpace = 0;
    in_buffer = new unsigned char[WRITE_CHUNK];
    out_buffer = new unsigned char[WRITE_CHUNK];
};


KSAFileWriter::~KSAFileWriter()
{
    delete[] in_buffer;
    delete[] out_buffer;
};

void KSAFileWriter::SetFileName(std::string filename)
{
    fFileName = filename;
    if(std::string::npos != filename.find(std::string(".ksa")))
    {
        fUseCompression = false; //plain text
    }
    else if(std::string::npos != filename.find(std::string(".zksa")))
    {
        //if we used the special extention we'll compress things
        fUseCompression = true; //compressed text
    }
    else
    {
        //default
        fUseCompression = false; //plain text
    }
};


bool
KSAFileWriter::Open()
{
    if(fUseCompression)
    {
        int ret;
        /* allocate deflate state */
        fZStream.zalloc = Z_NULL;
        fZStream.zfree = Z_NULL;
        fZStream.opaque = Z_NULL;
        ret = deflateInit(&fZStream, fCompressionLevel);
        if(ret != Z_OK){return false;};
        fUsedSpace = 0;

        if(fIsOpen)
        {
            fFileStream.close();
        }

        fFileStream.open(fFileName.c_str(), std::ofstream::trunc | std::ofstream::binary);
        //append mode not available for compressed files
        //possibily could add this feature in future
    }
    else
    {
        if(fIsOpen)
        {
            fFileStream.close();
        }

        //only allowed mode is recreate/truncate
        fFileStream.open(fFileName.c_str(), std::ofstream::trunc);
    }

    fIsOpen = fFileStream.is_open();
    return fIsOpen;
}

void
KSAFileWriter::AddToFile(const std::string& data)
{

    if(fUseCompression)
    {
        int size = data.size();
        int ret, flush, have;

        if(size >= WRITE_CHUNK)
        {
            //do no add this line to the file until we flush the input buffer

            //tell the z_stream how much we want to compress from the in_buffer
            fZStream.avail_in = fUsedSpace;
            fZStream.next_in = in_buffer;
            flush = Z_NO_FLUSH;

            /* run deflate() on input until output buffer not full, finish
               compression if all of source has been read in */
            do
            {
                fZStream.avail_out = WRITE_CHUNK;
                fZStream.next_out = out_buffer;
                ret = deflate(&fZStream, flush);    /* no bad return value */
                assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
                have = WRITE_CHUNK - fZStream.avail_out;

                fFileStream.write(reinterpret_cast<const char*>(out_buffer), have);
                if(have == 0){break;};
            }
            while (fZStream.avail_out == 0);
            assert(fZStream.avail_in == 0);     /* all input will be used */

            //reset the used space to zero
            fUsedSpace = 0;

            //line is too big for the input buffer so we'll have to compress it piece by piece
            int position = 0;
            int amount = 0;
            int total = 0;

            do
            {
                if( (size - total) > WRITE_CHUNK )
                {
                    amount = WRITE_CHUNK;
                    fUsedSpace = WRITE_CHUNK;
                }
                else
                {
                    amount = size - total;
                    fUsedSpace = amount;
                }

                total += amount;

                //stash a piece this line in the input buffer
                data.copy( reinterpret_cast<char*>( &(in_buffer[0]) ), amount, position);
                position += amount;

                //tell the z_stream how much we want to compress from the in_buffer
                fZStream.avail_in = fUsedSpace;
                fZStream.next_in = in_buffer;
                flush = Z_NO_FLUSH;

                /* run deflate() on input until output buffer not full, finish
                   compression if all of source has been read in */
                do
                {
                    fZStream.avail_out = WRITE_CHUNK;
                    fZStream.next_out = out_buffer;
                    ret = deflate(&fZStream, flush);    /* no bad return value */
                    assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
                    have = WRITE_CHUNK - fZStream.avail_out;
                    fFileStream.write(reinterpret_cast<const char*>(out_buffer), have);
                }
                while (fZStream.avail_out == 0);
                assert(fZStream.avail_in == 0);     /* all input will be used */

                //reset the used space to zero
                fUsedSpace = 0;

            }
            while( total < size );

        }
        else if( fUsedSpace + size >= WRITE_CHUNK)
        {
            //do no add this line to the file until we flush the input buffer

            //tell the z_stream how much we want to compress from the in_buffer
            fZStream.avail_in = fUsedSpace;
            fZStream.next_in = in_buffer;
            flush = Z_NO_FLUSH;

            /* run deflate() on input until output buffer not full, finish
               compression if all of source has been read in */
            do
            {
                fZStream.avail_out = WRITE_CHUNK;
                fZStream.next_out = out_buffer;
                ret = deflate(&fZStream, flush);    /* no bad return value */
                assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
                have = WRITE_CHUNK - fZStream.avail_out;
                fFileStream.write(reinterpret_cast<const char*>(out_buffer), have);
            }
            while (fZStream.avail_out == 0);
            assert(fZStream.avail_in == 0);     /* all input will be used */

            //reset the used space to zero
            fUsedSpace = 0;

            //stash this line in the input buffer
            data.copy( reinterpret_cast<char*>(&(in_buffer[fUsedSpace]) ), size, 0);
            fUsedSpace = fUsedSpace + size;
        }
        else
        {
            //stash this line in the input buffer
            data.copy( reinterpret_cast<char*>( &(in_buffer[fUsedSpace]) ), size, 0);
            fUsedSpace = fUsedSpace + size;
        }

    }
    else
    {
        if(fFileStream.good())
        {
            fFileStream << data;
        }
    }
}

void
KSAFileWriter::Close()
{

    int ret, flush, have;

    if(fUseCompression)
    {
        //finalize the compression

        //tell the z_stream how much we want to compress from the in_buffer
        fZStream.avail_in = fUsedSpace;
        fZStream.next_in = in_buffer;
        flush = Z_FINISH;

        /* run deflate() on input until output buffer not full, finish
           compression if all of source has been read in */

        do
        {
            fZStream.avail_out = WRITE_CHUNK;
            fZStream.next_out = out_buffer;
            ret = deflate(&fZStream, flush);    /* no bad return value */
            assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
            have = WRITE_CHUNK - fZStream.avail_out;
            fFileStream.write(reinterpret_cast<const char*>(out_buffer), have);
        }
        while (fZStream.avail_out == 0);
        assert(fZStream.avail_in == 0);     /* all input will be used */

        //reset the used space to zero
        fUsedSpace = 0;

        /* clean up and return */
        (void)deflateEnd(&fZStream);

        if(fIsOpen)
        {
            fFileStream.flush();
            fFileStream.close();
        }
    }
    else
    {
        if(fIsOpen)
        {
            fFileStream.flush();
            fFileStream.close();
        }
    }
}




}

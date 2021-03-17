#include "KEMKSAFileInterface.hh"

#include "KSAStructuredASCIIHeaders.hh"

using std::set;
using std::string;

namespace KEMField
{

void KEMKSAFileInterface::ReadKSAFile(KSAInputNode* node, const string& file_name, bool& result)
{
    result = false;
    set<string> fileList = KEMFileInterface::GetInstance()->CompleteFileList();

    std::string full_file_name = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + file_name;
    for (const auto& it : fileList) {
        if (it == full_file_name) {
            KSAFileReader reader;

            reader.SetFileName(full_file_name);
            if (reader.Open()) {
                KSAInputCollector collector;
                collector.SetFileReader(&reader);
                collector.ForwardInput(node);

                if (node->HasData()) {
                    result = true;
                }
            }
            return;
        }
    }
}


void KEMKSAFileInterface::SaveKSAFile(KSAOutputNode* node, const string& file_name, bool& result, bool forceOverwrite)
{
    result = false;
    set<string> fileList = KEMFileInterface::GetInstance()->CompleteFileList();
    std::string full_file_name = KEMFileInterface::GetInstance()->ActiveDirectory() + "/" + file_name;

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

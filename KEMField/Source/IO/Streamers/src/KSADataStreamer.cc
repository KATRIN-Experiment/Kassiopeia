#include "KSADataStreamer.hh"

namespace KEMField
{
void KSADataStreamer::open(const std::string& fileName, const std::string& action)
{
    std::string action_;
    action_.resize(action.length());

    std::transform(action.begin(), action.end(), action_.begin(), ::toupper);

    if (action_.compare("READ") == 0) {
        fIsReading = true;
        fReader.SetFileName(fileName);
        fReader.Open();

        // strip the first line
        std::string s;
        fReader.GetLine(s);
    }
    if (action_.compare("OVERWRITE") == 0) {
        fIsReading = false;
        fWriter.SetFileName(fileName);
        fWriter.Open();
    }
}

void KSADataStreamer::close()
{
    if (fIsReading)
        fReader.Close();
    else {
        if (fBuffer.StringifyBuffer().size() != 0)
            flush();
        fBuffer.FillBuffer("\n");
        flush();
        fWriter.Close();
    }
}

void KSADataStreamer::flush()
{
    if (fIsReading)
        return;
    fWriter.AddToFile(fBuffer.StringifyBuffer());
    fBuffer.Clear();
}
}  // namespace KEMField

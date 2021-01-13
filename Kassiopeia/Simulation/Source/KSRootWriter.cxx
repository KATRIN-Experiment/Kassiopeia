#include "KSRootWriter.h"

#include "KSException.h"
#include "KSWritersMessage.h"

namespace Kassiopeia
{

KSRootWriter::KSRootWriter() : fWriters(128) {}
KSRootWriter::KSRootWriter(const KSRootWriter& aCopy) : KSComponent(aCopy), fWriters(aCopy.fWriters) {}
KSRootWriter* KSRootWriter::Clone() const
{
    return new KSRootWriter(*this);
}
KSRootWriter::~KSRootWriter() = default;

void KSRootWriter::AddWriter(KSWriter* aWriter)
{
    if (fWriters.AddElement(aWriter) == -1) {
        wtrmsg(eError) << "<" << GetName() << "> could not add writer <" << aWriter->GetName() << ">" << eom;
        return;
    }
    wtrmsg_debug("<" << GetName() << "> adding writer <" << aWriter->GetName() << ">" << eom);
    return;
}
void KSRootWriter::RemoveWriter(KSWriter* aWriter)
{
    if (fWriters.RemoveElement(aWriter) == -1) {
        wtrmsg(eError) << "<" << GetName() << "> could not remove writer <" << aWriter->GetName() << ">" << eom;
        return;
    }
    wtrmsg_debug("<" << GetName() << "> removing writer <" << aWriter->GetName() << ">" << eom);
    return;
}

void KSRootWriter::ExecuteRun()
{
    try {
        for (int tIndex = 0; tIndex < fWriters.End(); tIndex++) {
            fWriters.ElementAt(tIndex)->ExecuteRun();
        }
    }
    catch (KSException const& e) {
        throw KSWriterError().Nest(e) << "Failed to write run data.";
    }
    return;
}
void KSRootWriter::ExecuteEvent()
{
    try {
        for (int tIndex = 0; tIndex < fWriters.End(); tIndex++) {
            fWriters.ElementAt(tIndex)->ExecuteEvent();
        }
    }
    catch (KSException const& e) {
        throw KSWriterError().Nest(e) << "Failed to write event data.";
    }
    return;
}
void KSRootWriter::ExecuteTrack()
{
    try {
        for (int tIndex = 0; tIndex < fWriters.End(); tIndex++) {
            fWriters.ElementAt(tIndex)->ExecuteTrack();
        }
    }
    catch (KSException const& e) {
        throw KSWriterError().Nest(e) << "Failed to write track data.";
    }
    return;
}
void KSRootWriter::ExecuteStep()
{
    try {
        for (int tIndex = 0; tIndex < fWriters.End(); tIndex++) {
            fWriters.ElementAt(tIndex)->ExecuteStep();
        }
    }
    catch (KSException const& e) {
        throw KSWriterError().Nest(e) << "Failed to write step data.";
    }
    return;
}

STATICINT sKSWriterDict = KSDictionary<KSRootWriter>::AddCommand(&KSRootWriter::AddWriter, &KSRootWriter::RemoveWriter,
                                                                 "add_writer", "remove_writer");

}  // namespace Kassiopeia

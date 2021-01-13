#include "KGGeometryPrinterBuilder.hh"

#include <KElementProcessor.hh>
#include <KRoot.h>

using namespace KGeoBag;
using namespace std;

namespace katrin
{

STATICINT sKGGeometryPrinterStructure = KGGeometryPrinterBuilder::Attribute<std::string>("name") +
                                        KGGeometryPrinterBuilder::Attribute<std::string>("file") +
                                        KGGeometryPrinterBuilder::Attribute<std::string>("path") +
                                        KGGeometryPrinterBuilder::Attribute<bool>("write_json") +
                                        KGGeometryPrinterBuilder::Attribute<bool>("write_xml") +
                                        KGGeometryPrinterBuilder::Attribute<std::string>("surfaces") +
                                        KGGeometryPrinterBuilder::Attribute<std::string>("spaces");

STATICINT sKGGeometryPrinter = KRootBuilder::ComplexElement<KGGeometryPrinter>("geometry_printer");

STATICINT sKGGeometryPrinterCompat = KElementProcessor::ComplexElement<KGGeometryPrinter>("geometry_printer");

}  // namespace katrin

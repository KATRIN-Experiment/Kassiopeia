#include "KSVTKGeneratorPainterBuilder.h"

#include "KVTKWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

STATICINT sKSVTKGeneratorPainterStructure = KSVTKGeneratorPainterBuilder::Attribute<std::string>("name") +
                                            KSVTKGeneratorPainterBuilder::Attribute<std::string>("path") +
                                            KSVTKGeneratorPainterBuilder::Attribute<std::string>("file") +
                                            KSVTKGeneratorPainterBuilder::Attribute<std::string>("magnetic_field") +
                                            KSVTKGeneratorPainterBuilder::Attribute<std::string>("electric_field") +
                                            KSVTKGeneratorPainterBuilder::Attribute<int>("num_samples") +
                                            KSVTKGeneratorPainterBuilder::Attribute<double>("scale_factor") +
                                            KSVTKGeneratorPainterBuilder::Attribute<std::string>("color_variable") +
                                            KSVTKGeneratorPainterBuilder::Attribute<std::string>("add_generator") +
                                            KSVTKGeneratorPainterBuilder::Attribute<KThreeVector>("add_color");

STATICINT sKSVTKGeneratorPainterWindow =
    KVTKWindowBuilder::ComplexElement<KSVTKGeneratorPainter>("vtk_generator_painter");

}  // namespace katrin

#include "KSVTKGeneratorPainterBuilder.h"

#include "KVTKWindowBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

using KGeoBag::KThreeVector;

STATICINT sKSVTKGeneratorPainterStructure = KSVTKGeneratorPainterBuilder::Attribute<string>("name") +
                                            KSVTKGeneratorPainterBuilder::Attribute<string>("path") +
                                            KSVTKGeneratorPainterBuilder::Attribute<string>("file") +
                                            KSVTKGeneratorPainterBuilder::Attribute<string>("magnetic_field") +
                                            KSVTKGeneratorPainterBuilder::Attribute<string>("electric_field") +
                                            KSVTKGeneratorPainterBuilder::Attribute<int>("num_samples") +
                                            KSVTKGeneratorPainterBuilder::Attribute<double>("scale_factor") +
                                            KSVTKGeneratorPainterBuilder::Attribute<string>("color_variable") +
                                            KSVTKGeneratorPainterBuilder::Attribute<string>("add_generator") +
                                            KSVTKGeneratorPainterBuilder::Attribute<KThreeVector>("add_color");

STATICINT sKSVTKGeneratorPainterWindow =
    KVTKWindowBuilder::ComplexElement<KSVTKGeneratorPainter>("vtk_generator_painter");

}  // namespace katrin

/*
 * KFMElectrostaticParametersBuilder.cc
 *
 *  Created on: 10 Aug 2015
 *      Author: wolfgang
 */

#include "KFMElectrostaticParametersBuilder.hh"

namespace katrin
{

template<> KFMElectrostaticParametersBuilder::~KComplexElement() = default;

STATICINT sKFMElectrostaticParametersStructure =
    KFMElectrostaticParametersBuilder::Attribute<std::string>("strategy") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("top_level_divisions") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("tree_level_divisions") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("expansion_degree") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("neighbor_order") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("maximum_tree_depth") +
    KFMElectrostaticParametersBuilder::Attribute<double>("region_expansion_factor") +
    KFMElectrostaticParametersBuilder::Attribute<bool>("use_region_size_estimation") +
    KFMElectrostaticParametersBuilder::Attribute<double>("world_cube_center_x") +
    KFMElectrostaticParametersBuilder::Attribute<double>("world_cube_center_y") +
    KFMElectrostaticParametersBuilder::Attribute<double>("world_cube_center_z") +
    KFMElectrostaticParametersBuilder::Attribute<double>("world_cube_length") +
    KFMElectrostaticParametersBuilder::Attribute<bool>("use_caching") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("verbosity") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("allowed_number") +
    KFMElectrostaticParametersBuilder::Attribute<unsigned int>("allowed_fraction") +
    KFMElectrostaticParametersBuilder::Attribute<double>("bias_degree") +
    KFMElectrostaticParametersBuilder::Attribute<double>("insertion_ratio");

} /* namespace katrin */

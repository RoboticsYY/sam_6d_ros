// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#define DEBUG_FLAG false

#include "custom_debug_node.hpp"
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <cstring>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>

using namespace TemplateExtension;

//! [op:ctor]
CustomDebugNode::CustomDebugNode(const ov::Output<ov::Node>& input) 
    : Op({input}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomDebugNode::validate_and_infer_types() {
    // Input: any shape tensor
    const auto& debug_input = input(0);
    auto input_shape = debug_input.get_partial_shape();
    auto elem_type = debug_input.get_element_type();
    
    // Output is the same as input
    set_output_type(0, elem_type, input_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomDebugNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<CustomDebugNode>(new_args.at(0));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomDebugNode::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CustomDebugNode::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (DEBUG_FLAG){
      std::cout << "======== [CPU custom_debug_node] ======== " << std::endl;
    }

    const auto& input_tensor = inputs[0];

    // Get input shape and data type
    auto input_shape = input_tensor.get_shape();
    auto elem_type = input_tensor.get_element_type();
    
    // Output is the same as input
    auto& out = outputs[0];
    out.set_shape(input_shape);
    
    // General data copy scheme - support all OpenVINO data types
    size_t total_elements = input_tensor.get_size();
    size_t element_size = elem_type.size();
    std::memcpy(out.data(), input_tensor.data(), total_elements * element_size);
    
    // Debug: record input data to file
    // const bool debug = true; // true / false
    if (DEBUG_FLAG) {
        std::cout << "[CustomDebugNode] OV custom_debug_node (input) shape: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << input_shape[i];
        }
        std::cout << "], dtype: " << elem_type.get_type_name() << std::endl;
        
        // Save to file, for comparison with PyTorch
        FILE* fp = fopen("output/ov_debug_node.txt", "a");
        if (fp) {
            fprintf(fp, "----- OV custom_debug_node (input) -----\n");
            fprintf(fp, "Shape: [");
            for (size_t i = 0; i < input_shape.size(); ++i) {
                if (i > 0) fprintf(fp, ", ");
                fprintf(fp, "%zu", input_shape[i]);
            }
            fprintf(fp, "], dtype: %s\n", elem_type.get_type_name().c_str());
            
            // Only keep the first MAX_PRINT_ELEMS data
            size_t max_elements = std::min(static_cast<size_t>(100), total_elements);
            
            // Format output according to data type
            if (elem_type == ov::element::f32) {
                const float* input_data = input_tensor.data<const float>();
                for (size_t i = 0; i < max_elements; ++i) {
                    fprintf(fp, "%.2f ", input_data[i]);
                }
            } else if (elem_type == ov::element::f16) {
                const ov::float16* input_data = input_tensor.data<const ov::float16>();
                for (size_t i = 0; i < max_elements; ++i) {
                    fprintf(fp, "%.2f ", static_cast<float>(input_data[i]));
                }
            } else if (elem_type == ov::element::i64) {
                const int64_t* input_data = input_tensor.data<const int64_t>();
                for (size_t i = 0; i < max_elements; ++i) {
                    fprintf(fp, "%ld ", input_data[i]);
                }
            } else if (elem_type == ov::element::i32) {
                const int32_t* input_data = input_tensor.data<const int32_t>();
                for (size_t i = 0; i < max_elements; ++i) {
                    fprintf(fp, "%d ", input_data[i]);
                }
            } else if (elem_type == ov::element::u8) {
                const uint8_t* input_data = input_tensor.data<const uint8_t>();
                for (size_t i = 0; i < max_elements; ++i) {
                    fprintf(fp, "%u ", input_data[i]);
                }
            } else if (elem_type == ov::element::boolean) {
                const bool* input_data = input_tensor.data<const bool>();
                for (size_t i = 0; i < max_elements; ++i) {
                    fprintf(fp, "%d ", input_data[i] ? 1 : 0);
                }
            } else {
                // For other data types, output raw byte data
                const uint8_t* raw_data = static_cast<const uint8_t*>(input_tensor.data());
                for (size_t i = 0; i < max_elements * element_size; ++i) {
                    fprintf(fp, "%02x ", raw_data[i]);
                }
            }
            
            // If data is truncated, add a note
            if (total_elements > 100) {
                fprintf(fp, "\n... (truncated, total %zu elements)", total_elements);
            }
            fprintf(fp, "\n");
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    return true;
}

bool CustomDebugNode::has_evaluate() const {
    return true;
}
//! [op:evaluate] 
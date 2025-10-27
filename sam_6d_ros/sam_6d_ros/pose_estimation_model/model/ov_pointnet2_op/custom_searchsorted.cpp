// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom_searchsorted.hpp"
#include <algorithm>
#include <stdexcept>

using namespace TemplateExtension;

//! [op:ctor]
CustomSearchSorted::CustomSearchSorted(const ov::Output<ov::Node>& cumsum_weights, const ov::Output<ov::Node>& random_values) 
    : Op({cumsum_weights, random_values}) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void CustomSearchSorted::validate_and_infer_types() {
    // Input 1: cumsum_weights shape: (batch_size, N)
    const auto& cumsum_input = input(0);
    auto cumsum_shape = cumsum_input.get_partial_shape();
    auto elem_type = get_input_element_type(0);
    
    // Input 2: random_values shape: (batch_size, num_samples)
    const auto& random_input = input(1);
    auto random_shape = random_input.get_partial_shape();
    
    // Validate inputs
    if (cumsum_shape.rank().is_static() && cumsum_shape.rank().get_length() == 2) {
        if (random_shape.rank().is_static() && random_shape.rank().get_length() == 2) {
            // Check if batch_size matches
            if (cumsum_shape[0].is_static() && random_shape[0].is_static()) {
                if (cumsum_shape[0] != random_shape[0]) {
                    throw std::runtime_error("Batch sizes must match between cumsum_weights and random_values");
                }
            }
            // Output shape: (batch_size, num_samples)
            set_output_type(0, ov::element::i64, ov::PartialShape{random_shape[0], random_shape[1]});
        } else {
            throw std::runtime_error("random_values must be a 2D tensor of shape (batch_size, num_samples)");
        }
    } else {
        throw std::runtime_error("cumsum_weights must be a 2D tensor of shape (batch_size, N)");
    }
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> CustomSearchSorted::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<CustomSearchSorted>(new_args.at(0), new_args.at(1));
}
//! [op:copy]

//! [op:visit_attributes]
bool CustomSearchSorted::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool CustomSearchSorted::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    const auto& cumsum_tensor = inputs[0];
    const auto& random_tensor = inputs[1];
    
    auto cumsum_shape = cumsum_tensor.get_shape();
    auto random_shape = random_tensor.get_shape();
    
    if (cumsum_shape.size() != 2 || random_shape.size() != 2) {
        throw std::runtime_error("Both inputs must be 2D tensors");
    }
    
    size_t batch_size = cumsum_shape[0];
    size_t N = cumsum_shape[1];
    size_t num_samples = random_shape[1];
    
    if (batch_size != random_shape[0]) {
        throw std::runtime_error("Batch sizes must match");
    }
    
    const float* cumsum_data = cumsum_tensor.data<const float>();
    const float* random_data = random_tensor.data<const float>();
    
    auto& out = outputs[0];
    out.set_shape({batch_size, num_samples});
    int64_t* out_data = out.data<int64_t>();
    
    // Implement searchsorted algorithm
    for (size_t b = 0; b < batch_size; ++b) {
        const float* batch_cumsum = cumsum_data + b * N;
        
        for (size_t s = 0; s < num_samples; ++s) {
            float random_val = random_data[b * num_samples + s];
            
            // Binary search to find the position of the first cumsum_weights >= random_values
            int64_t left = 0;
            int64_t right = N;
            
            while (left < right) {
                int64_t mid = left + (right - left) / 2;
                if (batch_cumsum[mid] >= random_val) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            
            out_data[b * num_samples + s] = left;
        }
    }
    
    // Debug: print CustomSearchSorted out_data
    const bool debug = false; // true / false
    if (debug) {
        std::cout << "[CustomSearchSorted Debug] First few indices: ";
        // for (size_t i = 0; i < std::min(size_t(10), batch_size * num_samples); ++i) {
        //     std::cout << out_data[i] << ' ';
        // }
        std::cout << std::endl;
        
        // Add input data debugging information
        std::cout << "[CustomSearchSorted Debug] Input shapes: cumsum(" << batch_size << "," << N << "), random(" << batch_size << "," << num_samples << ")" << std::endl;
        std::cout << "[CustomSearchSorted Debug] First batch cumsum values: ";
        for (size_t i = 0; i < std::min(size_t(10), N); ++i) {
            std::cout << cumsum_data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "[CustomSearchSorted Debug] First batch random values: ";
        for (size_t i = 0; i < std::min(size_t(10), num_samples); ++i) {
            std::cout << random_data[i] << " ";
        }
        std::cout << std::endl;
        
        // Save to file, for comparison with PyTorch
        FILE* fp = fopen("output/ov_searchsorted.txt", "a");
        if (fp) {
            fprintf(fp, "----- OV CustomSearchSorteds -----\n");
            for (size_t i = 0; i < batch_size * num_samples; ++i) {
                fprintf(fp, "%ld ", out_data[i]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }
    
    return true;
}

bool CustomSearchSorted::has_evaluate() const {
    return true;
}
//! [op:evaluate] 
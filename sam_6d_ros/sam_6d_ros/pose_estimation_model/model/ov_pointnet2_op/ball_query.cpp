// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ball_query.hpp"
#include <algorithm>
#include <omp.h>
#define DEBUG_FLAG false
using namespace TemplateExtension;

//! [op:ctor]
BallQuery::BallQuery(const ov::Output<ov::Node>& new_xyz, const ov::Output<ov::Node>& xyz, float radius_f, int nsample_i)
    : Op({new_xyz, xyz}), m_radius(radius_f), m_nsample(nsample_i) {
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void BallQuery::validate_and_infer_types() {
    if (get_input_size() != 2) {
        throw std::runtime_error("BallQuery expects 2 inputs (new_xyz, xyz), got " + std::to_string(get_input_size()));
    }
    const auto& new_xyz = input(0);
    // const auto& xyz = input(1);
    auto new_xyz_shape = new_xyz.get_partial_shape();
    ov::PartialShape output_shape = {new_xyz_shape[0], new_xyz_shape[1], m_nsample};
    set_output_type(0, ov::element::i32, output_shape);
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> BallQuery::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<BallQuery>(new_args.at(0), new_args.at(1), m_radius, m_nsample);
}
//! [op:copy]

//! [op:visit_attributes]
bool BallQuery::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("radius", m_radius);
    visitor.on_attribute("nsample", m_nsample);
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool BallQuery::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    if (DEBUG_FLAG){
      std::cout << "======== [CPU ov_ball_query] ======== " << std::endl;
    }
    // if (inputs.size() != 2) {
    //     throw std::runtime_error("BallQuery expects 2 inputs (new_xyz, xyz), got " + std::to_string(inputs.size()));
    // }
    // get attribute
    float radius = m_radius;
    int nsample = m_nsample;
    const float* new_xyz = inputs[0].data<const float>();
    const float* xyz = inputs[1].data<const float>();

    int b = inputs[1].get_shape()[0]; // batch size
    int n = inputs[1].get_shape()[1]; // number of points in xyz
    int npoint = inputs[0].get_shape()[1]; // number of points in new_xyz

    ov::PartialShape output_shape = {b, npoint, nsample};
    outputs[0].set_shape(output_shape.to_shape());
    auto& out_tensor = outputs[0];
    int *current_idx = out_tensor.data<int>();

    // Initialize output tensor to zeros (same as PyTorch torch::zeros)
    std::fill(current_idx, current_idx + b * npoint * nsample, 0);

    float radius2 = radius * radius;

    for (int batch_index = 0; batch_index < b; ++batch_index) {
      // Each batch's starting position
      const float *current_xyz = xyz + batch_index * n * 3;
      const float *current_new_xyz = new_xyz + batch_index * npoint * 3;
      int *current_batch_idx = current_idx + batch_index * npoint * nsample;

      for (int j = 0; j < npoint; ++j) {
        float new_x = current_new_xyz[j * 3 + 0];
        float new_y = current_new_xyz[j * 3 + 1];
        float new_z = current_new_xyz[j * 3 + 2];
        int cnt = 0;

        // Iterate over all original points to find points within the specified radius
        for (int k = 0; k < n && cnt < nsample; ++k) {
          float x = current_xyz[k * 3 + 0];
          float y = current_xyz[k * 3 + 1];
          float z = current_xyz[k * 3 + 2];
          float d2 = (new_x - x) * (new_x - x) +
                    (new_y - y) * (new_y - y) +
                    (new_z - z) * (new_z - z);

          if (d2 < radius2) {
            if (cnt == 0) {
              // Initialize index array, if not enough neighbors, repeat the last valid neighbor index
              for (int l = 0; l < nsample; ++l) {
                current_batch_idx[j * nsample + l] = k;
              }
            }
            current_batch_idx[j * nsample + cnt] = k;
            ++cnt;
          }
        }
      }
    }
    // Debug: print BallQuery out_tensor 
    // const bool debug = false; // true / false
    if (DEBUG_FLAG) {
        std::cout << "[BallQuery Debug] out_tensor: ";
        int total = b * npoint * nsample;
        int* out_data = out_tensor.data<int>();
        // for (int i = 0; i < total; ++i) {
        //     std::cout << out_data[i] << ' ';
        // }
        std::cout << std::endl;
        // Save to file, for comparison with PyTorch
        FILE* fp = fopen("output/ov_ball_query.txt", "a");
        if (fp) {
            fprintf(fp, "----- ball_query call -----\n");
            for (int i = 0; i < total; ++i) {
                fprintf(fp, "%d ", out_data[i]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        } else {
            std::cerr << "[BallQuery Debug] Failed to open output/ov_ball_query.txt for writing!" << std::endl;
        }
    }
    return true;
}

bool BallQuery::has_evaluate() const {
    return true;
}
//! [op:evaluate]